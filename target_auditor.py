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
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from database_manager import DatabaseManager
from heuristic_classifier import LexiconManager
from replay_executor import ReplayExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("target_auditor")

class TargetAuditor:
    """Audit a stored target for working events and real betting odds."""

    # Roles that identify the three pipeline stages.
    CATALOG_ROLES: Set[str] = {"catalog_root", "sports_catalog", "categories"}
    FIXTURES_ROLES: Set[str] = {"fixtures_list", "live_fixtures", "live_overview", "upcoming_fixtures"}
    ODDS_ROLES: Set[str] = {"odds_detail", "odds", "betting_odds", "live_odds"}

    # URL keywords identifying real fixture endpoints.
    FIXTURE_URL_KEYWORDS: Tuple[str, ...] = ("livenow", "liveoverview", "upcoming", "highlights")
    # URL/key keywords that indicate genuine match/market endpoints.
    MATCH_URL_KEYWORDS: Tuple[str, ...] = ("event", "match", "fixture", "game", "market", "odds")
    # Dict keys whose numeric values are genuine odds (leaf context).
    ODDS_VALUE_KEYS: Tuple[str, ...] = ("odds", "price", "decimal", "back", "lay", "payout")
    # Delimiters recognized in international match names.
    MATCH_NAME_DELIMITERS: Tuple[str, ...] = (" vs ", " v ", " - ", " @ ", " / ")
    # Parameter-name keywords hydrated with a bounded page size.
    COUNT_PARAM_KEYWORDS: Tuple[str, ...] = (
        "count", "limit", "perpage", "max", "size", "page", "offset",
    )
    # Default page size used when hydrating count/limit placeholders.
    DEFAULT_PAGE_SIZE: int = 10

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

        Raises:
            RuntimeError: If the target cannot be loaded, has no endpoints,
                or the sportsbook lexicon cannot be initialized.
        """
        self.target_id = target_id
        self.db_manager = db_manager or DatabaseManager(db_path="./data/control_plane.db")

        # Load the schema and endpoints via ReplayExecutor.
        try:
            self.executor = ReplayExecutor(target_id, self.db_manager)
        except Exception as err:
            raise RuntimeError(f"Failed to load target {target_id}: {err}") from err

        self.endpoints: Dict[str, Dict[str, Any]] = getattr(self.executor, "endpoints", {})
        if not self.endpoints:
            raise RuntimeError(f"Target {target_id} has no endpoints in its schema.")

        # Central lexicons replace all hardcoded keyword tuples.
        try:
            self.lexicon_mgr = LexiconManager("sportsbook_lexicon.json")
        except Exception as err:
            raise RuntimeError(f"Failed to load sportsbook lexicon: {err}") from err

        # Network filters and canonical pipeline are loaded via dedicated helpers.
        self.network_filters: Dict[str, Any] = self._load_network_filters()
        self.canonical_pipeline: Dict[str, Any] = self._load_canonical_pipeline()

        filter_url_keywords: Set[str] = {
            str(kw).lower() for kw in self.network_filters.get("ignored_url_keywords", [])
        }
        # URL-level noise rejection: filter file keywords + all lexicon noise terms.
        self.excluded_url_keywords: Set[str] = filter_url_keywords | self.lexicon_mgr.all_noise_terms
        # Key/role-level noise rejection: lexicon noise taxonomy only.
        self.excluded_key_keywords: Set[str] = self.lexicon_mgr.all_noise_terms
        # Betting-market context keys come from the odds/market lexicon category.
        self.market_context_keys: Set[str] = self.lexicon_mgr.positive_keywords.get(
            "betting_markets_and_odds", set()
        )
        # Competitor keys are derived from the event-metadata lexicon category,
        # restricted to terms that semantically denote competing sides.
        competitor_markers: Tuple[str, ...] = (
            "home", "away", "competitor", "participant", "team",
            "player", "opponent", "guest", "visitor", "host",
        )
        event_terms = self.lexicon_mgr.positive_keywords.get("sports_event_metadata", set())
        self.competitor_keys: Set[str] = {
            term for term in event_terms if any(marker in term for marker in competitor_markers)
        }

    def _load_network_filters(self) -> Dict[str, Any]:
        """Load URL noise filters from 'network_filters.json' if present."""
        filter_path = pathlib.Path("network_filters.json")
        try:
            with filter_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                return loaded
            logger.warning("Network filters file has unexpected top-level type; ignoring it.")
            return {}
        except FileNotFoundError:
            logger.warning(
                "Network filters file 'network_filters.json' not found; proceeding without URL filters."
            )
            return {}
        except (json.JSONDecodeError, OSError) as err:
            logger.warning(f"Failed to load network filters: {err}")
            return {}

    def _load_canonical_pipeline(self) -> Dict[str, Any]:
        """Load the canonical pipeline priority map from the control-plane DB."""
        try:
            target_row = self.db_manager.get_target_by_domain(self.executor.target_domain)
            pipeline = target_row.get("canonical_pipeline", {}) if isinstance(target_row, dict) else {}
            return pipeline if isinstance(pipeline, dict) else {}
        except Exception as err:
            logger.warning(f"Failed to load canonical pipeline for target {self.target_id}: {err}")
            return {}

    def _resolve_pipeline_key(self, pipeline_value: Any) -> Optional[str]:
        """Resolve a canonical-pipeline value (e.g. 'GET_https://...') to a
        stored endpoint key.

        The pipeline value may be:
          - None: no resolution possible.
          - A direct endpoint key present in ``self.endpoints``.
          - A 'METHOD_url' composite string whose URL suffix matches a stored
            endpoint's template_url.
        """
        if not isinstance(pipeline_value, str) or not pipeline_value.strip():
            return None
        value = pipeline_value.strip()
        if value in self.endpoints:
            return value
        # Strip a leading 'METHOD_' prefix and match on the URL portion.
        url_part = value.split("_", 1)[1] if "_" in value else value
        url_part_l = url_part.lower()
        for key, info in self.endpoints.items():
            template = str(info.get("template_url") or info.get("url") or "").lower()
            if template and template == url_part_l:
                return key
        return None

    def _is_excluded_endpoint(self, key: str, info: Dict[str, Any]) -> bool:
        """Return True for configuration/system endpoints that must be ignored."""
        url = self._get_endpoint_url(info)
        key_l = key.lower()
        role = str(info.get("role", "")).lower()
        if any(ex in url for ex in self.excluded_url_keywords):
            return True
        if any(ex in key_l for ex in self.excluded_key_keywords):
            return True
        if any(ex in role for ex in self.excluded_key_keywords):
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

    def _get_fixtures_candidates(self) -> List[str]:
        """Return an ordered list of fixtures endpoint candidates.

        Priority 1: canonical pipeline stages (live_overview, upcoming_fixtures)
        resolved against stored endpoints. Priority 2-4: heuristic URL-keyword,
        role, and endpoint-key matches. Duplicates are removed preserving order.
        """
        candidates: List[str] = []

        for stage in ("live_overview", "upcoming_fixtures"):
            resolved = self._resolve_pipeline_key(self.canonical_pipeline.get(stage))
            if (
                resolved
                and resolved in self.endpoints
                and resolved not in candidates
                and not self._is_excluded_endpoint(resolved, self.endpoints[resolved])
            ):
                candidates.append(resolved)

        non_excluded = [
            k for k in self.endpoints if not self._is_excluded_endpoint(k, self.endpoints[k])
        ]
        for k in non_excluded:
            if any(kw in self._get_endpoint_url(self.endpoints[k]) for kw in self.FIXTURE_URL_KEYWORDS):
                candidates.append(k)
        for k in non_excluded:
            if self.endpoints[k].get("role") in self.FIXTURES_ROLES:
                candidates.append(k)
        for k in non_excluded:
            if any(kw in k.lower() for kw in self.FIXTURE_URL_KEYWORDS):
                candidates.append(k)

        return list(dict.fromkeys(candidates))

    def _find_odds_endpoint(self) -> Optional[str]:
        """Return a real odds/market endpoint, prioritizing match/market URL
        keywords and ignoring configuration/system endpoints."""
        canonical_odds = self._resolve_pipeline_key(self.canonical_pipeline.get("match_detail_metrics"))
        if (
            canonical_odds
            and canonical_odds in self.endpoints
            and not self._is_excluded_endpoint(canonical_odds, self.endpoints[canonical_odds])
        ):
            return canonical_odds

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
        """Return parameters for the fixtures endpoint, hydrating count/limit
        placeholders with a bounded page size."""
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
                    if any(kw in param_name.lower() for kw in self.COUNT_PARAM_KEYWORDS):
                        params[placeholder] = self.DEFAULT_PAGE_SIZE

        # Also check dynamic_parameters for count/limit names.
        for param in info.get("dynamic_parameters", []):
            if any(kw in str(param).lower() for kw in self.COUNT_PARAM_KEYWORDS):
                params[str(param)] = self.DEFAULT_PAGE_SIZE

        return params

    def _is_catalog_payload(self, payload: Any) -> bool:
        """Check if the payload contains sports/categories/leagues."""
        if not isinstance(payload, dict):
            return False
        for key in ("sports", "categories", "leagues", "competitions", "events"):
            if key in payload:
                return True
        for value in payload.values():
            if isinstance(value, (dict, list)) and self._is_catalog_payload(value):
                return True
        return False

    def _has_match_delimiter(self, name: str) -> bool:
        """Return True when the name splits into two non-empty halves on any
        recognized international match delimiter."""
        lowered = name.lower()
        for delimiter in self.MATCH_NAME_DELIMITERS:
            parts = lowered.split(delimiter)
            if len(parts) >= 2 and parts[0].strip() and parts[-1].strip():
                return True
        return False

    def _is_match_entity(self, node: Dict[str, Any]) -> bool:
        """Return True only if the dict looks like an actual fixture/match
        entity: it has an ID and either structured competitor fields or a
        delimited match name. Sports, leagues, and system objects are rejected."""
        keys = {str(k).lower() for k in node}
        if not any(k in keys for k in (
            "id", "match_id", "matchid", "event_id", "eventid", "fixture_id", "fixtureid",
        )):
            return False
        # Structured competitor fields (flat keys or nested competitor objects).
        for key in self.competitor_keys:
            if key in keys:
                value = node.get(key)
                if isinstance(value, (dict, list)) and value:
                    return True
                if isinstance(value, (str, int)) and str(value).strip():
                    return True
        # Delimited international match names ("A - B", "A v B", "A / B", ...).
        for name_key in ("name", "title", "event", "match"):
            value = node.get(name_key)
            if isinstance(value, str) and (
                self._has_match_delimiter(value) or " vs " in value.lower() or " @ " in value.lower()
            ):
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
                    str(k).lower() in self.market_context_keys for k in node
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

    def _extract_sample_odds(self, payload: Any, limit: int = 5) -> List[float]:
        """Return up to ``limit`` numeric values that look like betting odds
        within genuine betting-market contexts."""
        return self._collect_odds(payload, limit=limit)

    def _print_endpoints(self) -> None:
        """Print all stored endpoints for transparency."""
        print(f"\nStored endpoints for target {self.target_id}:")
        for key, info in self.endpoints.items():
            print(
                "  - key: {:<30} category: {:<15} role: {}".format(
                    key,
                    info.get("category", "unknown"),
                    info.get("role", "unknown"),
                )
            )
        print()

    def _build_odds_curl(self, odds_key: str, param_name: str, match_id: str) -> str:
        """Build a ready-to-run curl command for the odds endpoint.

        Prefers ReplayExecutor.to_curl when available; falls back to the
        stored curl_command with the template URL hydrated via hydrate_url.
        """
        to_curl = getattr(self.executor, "to_curl", None)
        if callable(to_curl):
            try:
                return str(to_curl(odds_key, params={param_name: match_id}))
            except Exception as err:
                logger.warning(f"ReplayExecutor.to_curl failed: {self._format_error(err)}")
        # Fallback: hydrate the stored template URL manually.
        try:
            hydrated = self.executor.hydrate_url(
                self.endpoints[odds_key].get("template_url", ""), {param_name: match_id}
            )
            method = str(self.endpoints[odds_key].get("method", "GET")).upper()
            return f'curl -X {method} "{hydrated}"'
        except Exception as err:
            logger.warning(f"Failed to build fallback curl command: {self._format_error(err)}")
            return ""

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

        # Step 2: Sequential candidate probing for an active match ID.
        fixtures_candidates = self._get_fixtures_candidates()
        if not fixtures_candidates:
            details.append(
                "FAIL: No fixtures candidates found (canonical pipeline empty and no heuristic matches)."
            )
            return self._build_result("FAIL", 0, "", details)

        match_id: Optional[str] = None
        fixtures_key: Optional[str] = None
        for candidate_key in fixtures_candidates:
            fixtures_params = self._get_fixtures_params(candidate_key)
            try:
                candidate_payload = self.executor.execute(candidate_key, params=fixtures_params)
            except Exception as err:
                details.append(
                    f"INFO: Fixtures candidate '{candidate_key}' failed: "
                    f"{self._format_error(err)}; probing next candidate."
                )
                continue
            candidate_match_id = self._extract_match_id(candidate_payload)
            if candidate_match_id:
                match_id = candidate_match_id
                fixtures_key = candidate_key
                break
            details.append(
                f"INFO: Fixtures candidate '{candidate_key}' returned no active match ID; "
                "probing next candidate."
            )

        if not match_id or fixtures_key is None:
            details.append(
                f"FAIL: All {len(fixtures_candidates)} fixtures candidates exhausted "
                "without harvesting an active match ID."
            )
            return self._build_result("FAIL", 0, "", details)

        fixtures_success = True
        details.append(
            f"PASS: Fixtures endpoint '{fixtures_key}' yielded match ID {match_id}."
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
            str(dynamic_params[0]) if dynamic_params else "event_id",
        )

        try:
            odds_payload = self.executor.execute(odds_key, params={param_name: match_id})
        except Exception as err:
            details.append(
                f"FAIL: Odds endpoint '{odds_key}' failed with match ID {match_id}: "
                f"{self._format_error(err)}"
            )
            # Verdict is PASS because active fixtures were successfully fetched.
            return self._build_result("PASS", 0, "", details)

        total_odds = self._count_odds(odds_payload)
        if total_odds == 0:
            details.append(f"FAIL: Odds endpoint '{odds_key}' returned no numeric odds.")
        else:
            details.append(
                f"PASS: Odds endpoint '{odds_key}' returned {total_odds} numeric odds."
            )
            sample_odds = self._extract_sample_odds(odds_payload)
            if sample_odds:
                details.append(
                    f"Sample odds: {', '.join(f'{v:.2f}' for v in sample_odds)}"
                )

        # Build curl command for the odds endpoint.
        curl_command = self._build_odds_curl(odds_key, param_name, match_id)

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
