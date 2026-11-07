import dataclasses
import json
import sys
from typing import Any, Dict, List
from odds_normalizer import OddsNormalizer
from replay_executor import ReplayExecutor


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_norm.py <target_domain> [endpoint_filter]")
        print("Example: python test_norm.py playdoit.mx")
        print("Example: python test_norm.py playdoit.mx GetUpcoming")
        sys.exit(1)

    target = sys.argv[1]
    endpoint_filter = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        executor = ReplayExecutor(target)
    except Exception as err:
        print(f"[-] Error initializing ReplayExecutor for {target}: {err}")
        sys.exit(1)

    if not executor.endpoints:
        print(f"[-] No endpoints registered for {target}")
        sys.exit(1)

    # Filter endpoints if a filter was specified
    selected_endpoints: Dict[str, Any] = {}
    for key, info in executor.endpoints.items():
        if endpoint_filter is None or endpoint_filter.lower() in key.lower():
            selected_endpoints[key] = info

    if not selected_endpoints:
        print(f"[-] No endpoints match filter '{endpoint_filter}'.")
        print(f"[*] Available endpoints:\n  - " + "\n  - ".join(executor.endpoints.keys()))
        sys.exit(1)

    normalizer = OddsNormalizer()
    field_mappings = getattr(executor, "schema_data", {}).get("field_mappings", {})

    print(f"\n=======================================================")
    print(f"       NORMALIZATION DIAGNOSTIC TEST: {target}          ")
    print(f"=======================================================\n")
    print(f"[*] Total endpoints to evaluate: {len(selected_endpoints)}\n")

    success_count = 0
    failure_count = 0
    first_preview_shown = False

    for idx, (ep_key, ep_info) in enumerate(selected_endpoints.items(), start=1):
        print(f"[{idx}/{len(selected_endpoints)}] Testing: {ep_key}")
        try:
            raw_data = executor.execute(ep_key)
            normalized = normalizer.normalize(raw_data, field_mappings=field_mappings)

            # Extract metrics
            event_count = len(normalized) if isinstance(normalized, list) else (1 if normalized else 0)
            market_count = 0
            odds_count = 0

            events_list: List[Any] = normalized if isinstance(normalized, list) else ([normalized] if normalized else [])
            for ev in events_list:
                markets = getattr(ev, "markets", [])
                market_count += len(markets)
                for mkt in markets:
                    odds_count += len(getattr(mkt, "outcomes", []))

            print(f"    --> [OK 200] Events: {event_count} | Markets: {market_count} | Odds: {odds_count}")
            success_count += 1

            # Show preview only for the first endpoint yielding real markets
            if market_count > 0 and not first_preview_shown:
                first_preview_shown = True
                clean_preview = [to_dict(x) for x in events_list[:1]]
                print("\n    [+] Sample of first successfully normalized event:")
                formatted = json.dumps(clean_preview, indent=2, ensure_ascii=False)
                lines = formatted.splitlines()[:35]
                print("    " + "\n    ".join(lines))
                if len(formatted.splitlines()) > 35:
                    print("    ... [remaining output truncated for readability]\n")

        except Exception as err:
            failure_count += 1
            print(f"    --> [FAILED] Error: {err}")

    print("\n=======================================================")
    print(f"  SUMMARY: {success_count} Succeeded | {failure_count} Failed of {len(selected_endpoints)}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()