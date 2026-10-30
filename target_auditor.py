#!/usr/bin/env python3
"""
target_auditor.py – Audit a stored target for working events and real betting odds.

The auditor runs a three-step pipeline:
  1. Catalog endpoint returns sports/categories.
  2. Live fixtures endpoint returns an active match ID.
  3. Odds endpoint (using that match ID) returns numeric betting odds.

Usage:
    python target_auditor.py
    python target_auditor.py --target-id 1
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from replay_executor import ReplayExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("target_auditor")


class TargetAuditor:
    """Audit a stored target for working events and real betting odds."""

    # Roles that identify the three pipeline stages.
    CATALOG_ROLES = {"catalog_root", "sports_catalog", "categories"}
    FIXTURES_ROLES = {"fixtures_list", "live_fixtures", "live_overview", "upcoming_fixtures"}
    ODDS_ROLES = {"odds_detail", "odds", "betting_odds", "live_odds"}

    def __init__(
        self,
        target_id: int,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        """
        Initialize the auditor for a specific target.

        Args:
            target_id: Database ID of the target to audit.
            db_manager: Optional DatabaseManager instance. If None, a default
                manager is created using the standard control-plane DB path.
        """
        self.target_id = target_id
        self.db_manager = db_manager or DatabaseManager(db_path="./data/control_plane.db")

        # Load the schema and endpoints via ReplayExecutor.
        try:
            self.executor = ReplayExecutor(target_id, self.db_manager)
        except Exception as err:
            raise RuntimeError(f"Failed to load target {target_id}: {err}") from err

        self.endpoints: Dict[str, Dict[str, Any]] = self.executor.endpoints
        if not self.endpoints:
            raise RuntimeError(f"Target {target_id} has no endpoints in its schema.")

    def _find_endpoint_by_roles(self, roles: Set[str]) -> Optional[str]:
        """Return the first endpoint key whose role is in ``roles``."""
        for key, info in self.endpoints.items():
            if info.get("role", "").lower() in roles:
                return key
        return None

    def _is_catalog_payload(self, payload: Any) -> bool:
        """Check if the payload contains sports/categories/leagues."""
        if not isinstance(payload, dict):
            return False
        # Look for common keys that indicate a catalog.
        for key in ("sports", "categories", "leagues", "competitions", "events"):
            if key in payload:
                return True
        # Recursively check nested dicts/lists.
        for value in payload.values():
            if isinstance(value, (dict, list)):
                if self._is_catalog_payload(value):
                    return True
        return False

    def _extract_match_id(self, payload: Any) -> Optional[str]:
        """Recursively search for a numeric ID that looks like a match/event ID."""
        if isinstance(payload, dict):
            # Prefer keys that explicitly indicate a match/event.
            for key in ("match_id", "event_id", "fixture_id", "id"):
                if key in payload:
                    val = payload[key]
                    if isinstance(val, (int, str)) and not isinstance(val, bool):
                        return str(val)
            # Recurse into values.
            for value in payload.values():
                result = self._extract_match_id(value)
                if result is not None:
                    return result
        elif isinstance(payload, list):
            for item in payload:
                result = self._extract_match_id(item)
                if result is not None:
                    return result
        return None

    def _count_odds(self, payload: Any) -> int:
        """Count numeric values that look like betting odds (>1.0 and <1000.0)."""
        count = 0
        if isinstance(payload, dict):
            for value in payload.values():
                count += self._count_odds(value)
        elif isinstance(payload, list):
            for item in payload:
                count += self._count_odds(item)
        elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
            # Odds are typically >1.0 and <1000.0; ignore IDs and small integers.
            if 1.0 < float(payload) < 1000.0:
                count += 1
        return count

    def audit(self) -> Dict[str, Any]:
        """
        Run the three-step pipeline audit.

        Returns:
            A dictionary with:
                - verdict: "PASS" or "FAIL"
                - total_odds: number of numeric odds found
                - curl_command: ready-to-run curl for the odds endpoint
                - details: human-readable summary of each step
        """
        details: List[str] = []
        total_odds = 0
        curl_command = ""

        # Step 1: Catalog endpoint
        catalog_key = self._find_endpoint_by_roles(self.CATALOG_ROLES)
        if not catalog_key:
            details.append("FAIL: No catalog endpoint found (role in {catalog_root, sports_catalog, categories}).")
            return self._build_result("FAIL", 0, "", details)

        try:
            catalog_payload = self.executor.execute(catalog_key)
        except Exception as err:
            details.append(f"FAIL: Catalog endpoint '{catalog_key}' failed: {err}")
            return self._build_result("FAIL", 0, "", details)

        if not self._is_catalog_payload(catalog_payload):
            details.append(f"FAIL: Catalog endpoint '{catalog_key}' did not return sports/categories.")
            return self._build_result("FAIL", 0, "", details)
        details.append(f"PASS: Catalog endpoint '{catalog_key}' returned sports/categories.")

        # Step 2: Live fixtures endpoint
        fixtures_key = self._find_endpoint_by_roles(self.FIXTURES_ROLES)
        if not fixtures_key:
            details.append("FAIL: No live fixtures endpoint found (role in {fixtures_list, live_fixtures, live_overview, upcoming_fixtures}).")
            return self._build_result("FAIL", 0, "", details)

        try:
            fixtures_payload = self.executor.execute(fixtures_key)
        except Exception as err:
            details.append(f"FAIL: Live fixtures endpoint '{fixtures_key}' failed: {err}")
            return self._build_result("FAIL", 0, "", details)

        match_id = self._extract_match_id(fixtures_payload)
        if not match_id:
            details.append(f"FAIL: Live fixtures endpoint '{fixtures_key}' did not yield a match ID.")
            return self._build_result("FAIL", 0, "", details)
        details.append(f"PASS: Live fixtures endpoint '{fixtures_key}' yielded match ID {match_id}.")

        # Step 3: Odds endpoint
        odds_key = self._find_endpoint_by_roles(self.ODDS_ROLES)
        if not odds_key:
            details.append("FAIL: No odds endpoint found (role in {odds_detail, odds, betting_odds, live_odds}).")
            return self._build_result("FAIL", 0, "", details)

        # Determine the parameter name for the match ID.
        odds_info = self.endpoints[odds_key]
        dynamic_params = odds_info.get("dynamic_parameters", [])
        param_name = dynamic_params[0] if dynamic_params else "event_id"

        try:
            odds_payload = self.executor.execute(odds_key, params={param_name: match_id})
        except Exception as err:
            details.append(f"FAIL: Odds endpoint '{odds_key}' failed with match ID {match_id}: {err}")
            return self._build_result("FAIL", 0, "", details)

        total_odds = self._count_odds(odds_payload)
        if total_odds == 0:
            details.append(f"FAIL: Odds endpoint '{odds_key}' returned no numeric odds.")
            return self._build_result("FAIL", 0, "", details)

        details.append(f"PASS: Odds endpoint '{odds_key}' returned {total_odds} numeric odds.")

        # Build curl command for the odds endpoint.
        try:
            curl_command = self.executor.to_curl(odds_key, params={param_name: match_id})
        except Exception as err:
            logger.warning(f"Failed to generate curl command: {err}")
            curl_command = ""

        return self._build_result("PASS", total_odds, curl_command, details)

    def _build_result(
        self,
        verdict: str,
        total_odds: int,
        curl_command: str,
        details: List[str],
    ) -> Dict[str, Any]:
        """Assemble the final result dictionary."""
        return {
            "verdict": verdict,
            "total_odds": total_odds,
            "curl_command": curl_command,
            "details": details,
        }


def list_targets(db_manager: DatabaseManager) -> List[Dict[str, Any]]:
    """Return a list of all targets from the database."""
    conn = db_manager.get_connection()
    rows = conn.execute("SELECT id, domain, provider_family FROM targets ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def run_cli() -> None:
    """Interactive CLI to select and audit a target."""
    parser = argparse.ArgumentParser(description="Audit a stored target for working events and odds.")
    parser.add_argument("--target-id", type=int, help="Database ID of the target to audit.")
    args = parser.parse_args()

    db_manager = DatabaseManager(db_path="./data/control_plane.db")
    targets = list_targets(db_manager)

    if not targets:
        print("[-] No targets found in the database. Run discovery first.")
        sys.exit(1)

    if args.target_id is not None:
        target_id = args.target_id
        if not any(t["id"] == target_id for t in targets):
            print(f"[-] Target ID {target_id} not found.")
            sys.exit(1)
    else:
        print("\nAvailable targets:")
        for idx, t in enumerate(targets, start=1):
            print(f"  [{idx}] ID={t['id']}  Domain={t['domain']}  Provider={t['provider_family']}")
        while True:
            try:
                choice = int(input("\nSelect target number: ").strip())
                if 1 <= choice <= len(targets):
                    target_id = targets[choice - 1]["id"]
                    break
                print("[-] Invalid selection.")
            except (ValueError, EOFError):
                print("[-] Invalid input.")

    print(f"\n[*] Auditing target ID {target_id}...")
    try:
        auditor = TargetAuditor(target_id, db_manager)
        result = auditor.audit()
    except Exception as err:
        print(f"[-] Audit failed: {err}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"VERDICT: {result['verdict']}")
    print(f"TOTAL ODDS: {result['total_odds']}")
    if result["curl_command"]:
        print(f"CURL COMMAND:\n  {result['curl_command']}")
    print("\nDetails:")
    for line in result["details"]:
        print(f"  {line}")
    print("=" * 60)


if __name__ == "__main__":
    run_cli()
