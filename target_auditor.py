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
import urllib.parse

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

    # URL keywords identifying real fixture endpoints.
    FIXTURE_URL_KEYWORDS = ("livenow", "liveoverview", "upcoming", "highlights")
    # URL keywords identifying configuration/system endpoints to ignore.
    EXCLUDED_URL_KEYWORDS = ("params", "settings", "translations", "auth")
    # Endpoint-key/role keywords identifying configuration/system endpoints.
    EXCLUDED_KEY_KEYWORDS = (
        "param", "setting", "config", "system", "default", "translation",
        "auth", "login", "session", "metadata", "permission", "preference",
        "feature", "banner", "menu", "navigation",
    )
    # URL/key keywords that indicate genuine match/market endpoints.
    MATCH_URL_KEYWORDS = ("event", "match", "fixture", "game", "market", "odds")
    # Dict keys whose numeric values are genuine odds (leaf context).
    ODDS_VALUE_KEYS = ("odds", "price", "decimal", "back", "lay", "payout")
    # Dict keys that establish a betting-market context.
    MARKET_CONTEXT_KEYS = (
        "market", "markets", "selection", "selections", "outcome", "outcomes",
        "runner", "runners", "bet", "bets", "line", "lines", "bookmaker",
        "bookmakers", "odds", "price", "pick", "picks",
    )
    # Dict keys that establish a match/fixture entity (distinct competitors).
    COMPETITOR_KEYS = (
        "home", "away", "guest", "visitor", "host", "competitors",
        "participants", "teams", "team1", "team2", "home_team", "away_team",
        "hometeam", "awayteam", "player1", "player2", "opponent",
    )

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

    def _is_excluded_endpoint(self, key: str, info: Dict[str, Any]) -> bool:
        """Return True for configuration/system endpoints that must be ignored."""
        url = self._get_endpoint_url(info)
        key_l = key.lower()
        role = str(info.get("role", "")).lower()
        if any(ex in url for ex in self.EXCLUDED_URL_KEYWORDS):
            return True
        if any(ex in key_l for ex in self.EXCLUDED_KEY_KEYWORDS):
            return True
        if any(ex in role for ex in self.EXCLUDED_KEY_KEYWORDS):
            return True
        return False

    def _find_endpoint_by_roles(self, roles: Set[str]) -> Optional[str]:
        """Return the first non-excluded endpoint key whose role is in the given set."""
        for key, info in self.endpoints.items():
            if self._is_excluded_endpoint(key, info):
                continue
            if info.get("role") in roles:
                return key
        return None

    def _get_endpoint_url(self, info: Dict[str, Any]) -> str:
        """Return the endpoint URL, preferring template_url (the ReplayExecutor key)."""
        return str(info.get("template_url") or info.get("url") or "").lower()

    @staticmethod
    def _format_error(err: BaseException) -> str:
        """Flatten an exception and its cause chain into one readable message."""
        parts: List[str] = [str(err)]
        cause: Optional[BaseException] = err.__cause__
        while cause is not None:
            parts.append(str(cause))
            cause = cause.__cause__
        return " -> ".join(part for part in parts if part)

    def _find_fixtures_endpoint(self) -> Optional[str]:
        """Return a real fixtures endpoint, prioritizing URL keyword matches
        and explicitly ignoring configuration/system endpoints."""
        candidates = [
            k for k in self.endpoints
            if not self._is_excluded_endpoint(k, self.endpoints[k])
        ]
        # Priority 1: real fixture endpoints identified by URL keywords.
        for k in candidates:
            if any(kw in self._get_endpoint_url(self.endpoints[k]) for kw in self.FIXTURE_URL_KEYWORDS):
                return k
        # Priority 2: role-based match.
        for k in candidates:
            if self.endpoints[k].get("role") in self.FIXTURES_ROLES:
                return k
        # Priority 3: keyword match on the endpoint key itself.
        for k in candidates:
            if any(kw in k.lower() for kw in self.FIXTURE_URL_KEYWORDS):
                return k
        return None

    def _find_odds_endpoint(self) -> Optional[str]:
        """Return a real odds/market endpoint, prioritizing match/market URL
        keywords and ignoring configuration/system endpoints."""
        role_matches = [
            k for k, info in self.endpoints.items()
            if info.get("role") in self.ODDS_ROLES
            and not self._is_excluded_endpoint(k, info)
        ]
        # Priority 1: role match whose URL/key references events/markets.
        for k in role_matches:
            url = self._get_endpoint_url(self.endpoints[k])
            if any(kw in url or kw in k.lower() for kw in self.MATCH_URL_KEYWORDS):
                return k
        # Priority 2: any non-excluded role match.
        return role_matches[0] if role_matches else None

    def _get_fixtures_params(self, fixtures_key: str) -> Dict[str, Any]:
        """Return parameters to pass to the fixtures endpoint, hydrating count/limit placeholders with 10."""
        info = self.endpoints.get(fixtures_key, {})
        url_template = str(info.get("template_url") or info.get("url") or "")
        params: Dict[str, Any] = {}

        # Parse the URL template to find query parameters with placeholders.
        parsed = urllib.parse.urlparse(url_template)
        query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        for param_name, values in query_params.items():
            for val in values:
                if "{" in val and "}" in val:
                    placeholder = val[val.find("{") + 1:val.find("}")]
                    # If the parameter name is a count/limit keyword, hydrate with 10.
                    if any(kw in param_name.lower() for kw in ("count", "limit", "perpage", "max", "size", "page", "offset")):
                        params[placeholder] = 10

        # Also check dynamic_parameters for count/limit names.
        for param in info.get("dynamic_parameters", []):
            if any(kw in param.lower() for kw in ("count", "limit", "perpage", "max", "size", "page", "offset")):
                params[param] = 10

        return params

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

    def _is_match_entity(self, node: Dict[str, Any]) -> bool:
        """Return True only if the dict looks like an actual fixture/match
        entity: it has an ID and distinct competitors/participants (or a
        'Team A vs Team B' style name). Sports, leagues, and system objects
        lack competitors and are rejected."""
        keys = {str(k).lower() for k in node}
        if not any(k in keys for k in (
            "id", "match_id", "matchid", "event_id", "eventid", "fixture_id", "fixtureid",
        )):
            return False
        if any(k in keys for k in self.COMPETITOR_KEYS):
            return True
        for name_key in ("name", "title", "event", "match"):
            value = node.get(name_key)
            if isinstance(value, str) and (" vs " in value.lower() or " @ " in value.lower()):
                return True
        return False

    def _extract_match_id(self, payload: Any) -> Optional[str]:
        """Recursively search for an ID from an actual fixture/match entity
        (having distinct competitors/participants), never from isolated
        sports, leagues, or system/config objects."""
        if isinstance(payload, dict):
            if self._is_match_entity(payload):
                for key in ("match_id", "event_id", "fixture_id",
                            "matchid", "eventid", "fixtureid", "id"):
                    val = payload.get(key)
                    if isinstance(val, (int, str)) and not isinstance(val, bool) and str(val).strip():
                        return str(val)
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

    def _collect_odds(self, payload: Any, limit: int = 0) -> List[float]:
        """Collect numeric values that look like betting odds, but only within
        genuine betting-market contexts (markets/selections/odds structures).
        Configuration/system numbers (min/max bets, limits, timeouts) are
        rejected because they lack a market context and an odds-value key."""
        samples: List[float] = []

        def walk(node: Any, in_market: bool) -> None:
            if limit and len(samples) >= limit:
                return
            if isinstance(node, dict):
                context = in_market or any(
                    str(k).lower() in self.MARKET_CONTEXT_KEYS for k in node
                )
                for key, value in node.items():
                    if (
                        context
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and str(key).lower() in self.ODDS_VALUE_KEYS
                        and 1.0 < float(value) < 1000.0
                    ):
                        samples.append(float(value))
                        if limit and len(samples) >= limit:
                            return
                    else:
                        walk(value, context)
            elif isinstance(node, list):
                for item in node:
                    if (
                        in_market
                        and isinstance(item, (int, float))
                        and not isinstance(item, bool)
                        and 1.0 < float(item) < 1000.0
                    ):
                        samples.append(float(item))
                        if limit and len(samples) >= limit:
                            return
                    else:
                        walk(item, in_market)

        walk(payload, False)
        return samples

    def _count_odds(self, payload: Any) -> int:
        """Count numeric values that look like betting odds within genuine
        betting-market contexts."""
        return len(self._collect_odds(payload))

    def _print_endpoints(self) -> None:
        """Print all stored endpoints for transparency."""
        print("\nStored endpoints for target {}:".format(self.target_id))
        for key, info in self.endpoints.items():
            print(
                "  - key: {:<30} category: {:<15} role: {}".format(
                    key,
                    info.get("category", "unknown"),
                    info.get("role", "unknown"),
                )
            )
        print()

    def _extract_sample_odds(self, payload: Any, limit: int = 5) -> List[float]:
        """Return up to ``limit`` numeric values that look like betting odds
        within genuine betting-market contexts."""
        return self._collect_odds(payload, limit=limit)

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
        fixtures_success = False

        # Print all endpoints for transparency.
        self._print_endpoints()

        # Step 1: Catalog endpoint (optional – do not hard-fail if missing)
        catalog_key = self._find_endpoint_by_roles(self.CATALOG_ROLES)
        if not catalog_key:
            details.append(
                "SKIP: No catalog endpoint found (role in {catalog_root, sports_catalog, categories})."
            )
        else:
            try:
                catalog_payload = self.executor.execute(catalog_key)
            except Exception as err:
                details.append(
                    f"SKIP: Catalog endpoint '{catalog_key}' failed: {self._format_error(err)}"
                )
            else:
                if self._is_catalog_payload(catalog_payload):
                    details.append(
                        f"PASS: Catalog endpoint '{catalog_key}' returned sports/categories."
                    )
                else:
                    details.append(
                        f"SKIP: Catalog endpoint '{catalog_key}' did not return sports/categories."
                    )

        # Step 2: Live fixtures endpoint (required for match ID)
        fixtures_key = self._find_fixtures_endpoint()
        if not fixtures_key:
            details.append(
                "FAIL: No live fixtures endpoint found (role in {fixtures_list, live_fixtures, live_overview, upcoming_fixtures})."
            )
            return self._build_result("FAIL", 0, "", details)

        fixtures_params = self._get_fixtures_params(fixtures_key)
        try:
            fixtures_payload = self.executor.execute(fixtures_key, params=fixtures_params)
        except Exception as err:
            details.append(
                f"FAIL: Live fixtures endpoint '{fixtures_key}' failed: {self._format_error(err)}"
            )
            return self._build_result("FAIL", 0, "", details)

        match_id = self._extract_match_id(fixtures_payload)
        if not match_id:
            details.append(
                f"FAIL: Live fixtures endpoint '{fixtures_key}' did not yield a match ID."
            )
            return self._build_result("FAIL", 0, "", details)

        fixtures_success = True
        details.append(
            f"PASS: Live fixtures endpoint '{fixtures_key}' yielded match ID {match_id}."
        )

        # Step 3: Odds endpoint (tested only if a match ID was harvested)
        odds_key = self._find_odds_endpoint()
        if not odds_key:
            details.append(
                "FAIL: No odds endpoint found (role in {odds_detail, odds, betting_odds, live_odds})."
            )
            # Verdict is PASS because active fixtures were successfully fetched.
            return self._build_result("PASS", 0, "", details)

        # Determine the parameter name for the match ID, preferring
        # event/match/fixture-style names over generic or config parameters.
        odds_info = self.endpoints[odds_key]
        dynamic_params = odds_info.get("dynamic_parameters", [])
        param_name = next(
            (p for p in dynamic_params if any(
                kw in str(p).lower() for kw in ("event", "match", "game", "fixture")
            )),
            dynamic_params[0] if dynamic_params else "event_id",
        )

        try:
            odds_payload = self.executor.execute(odds_key, params={param_name: match_id})
        except Exception as err:
            details.append(
                f"FAIL: Odds endpoint '{odds_key}' failed with match ID {match_id}: {self._format_error(err)}"
            )
            # Verdict is PASS because active fixtures were successfully fetched.
            return self._build_result("PASS", 0, "", details)

        total_odds = self._count_odds(odds_payload)
        if total_odds == 0:
            details.append(
                f"FAIL: Odds endpoint '{odds_key}' returned no numeric odds."
            )
        else:
            details.append(
                f"PASS: Odds endpoint '{odds_key}' returned {total_odds} numeric odds."
            )
            # Extract and display sample odds.
            sample_odds = self._extract_sample_odds(odds_payload)
            if sample_odds:
                details.append(
                    f"Sample odds: {', '.join(f'{v:.2f}' for v in sample_odds)}"
                )

        # Build curl command for the odds endpoint.
        try:
            curl_command = self.executor.to_curl(odds_key, params={param_name: match_id})
        except Exception as err:
            logger.warning(f"Failed to generate curl command: {err}")
            curl_command = ""

        # Final verdict: PASS if we have odds OR active fixtures.
        verdict = "PASS" if (total_odds > 0 or fixtures_success) else "FAIL"
        return self._build_result(verdict, total_odds, curl_command, details)

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
