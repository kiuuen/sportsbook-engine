"""Post-discovery noise assistant for residual third-party traffic.

Analyzes captured network events for third-party telemetry/noise domains,
excludes hosts that are immune (target domain or hosts of positively
classified endpoints), and interactively offers to persist the detected
noise domains into ``network_filters.json``.
"""

import json
import logging
import pathlib
import urllib.parse
from typing import Any, Dict, List, Set

from cdp_interceptor import NetworkEvent
from heuristic_classifier.models import ClassificationResult


class NoiseAssistant:
    """Detects residual noise domains and updates the ignore filter list."""

    @staticmethod
    def process_noise_domains(
        events: List[NetworkEvent],
        classified_results: List[ClassificationResult],
        target_url: str,
        filters_path: str = "network_filters.json",
    ) -> None:
        """Aggregate residual third-party domains and offer to ignore them.

        Args:
            events: All captured network events from the discovery session.
            classified_results: Classification outcomes used to derive immune hosts.
            target_url: The target sportsbook URL whose domain is immune.
            filters_path: Path to the JSON filters file to update.
        """
        target_host = urllib.parse.urlparse(target_url).netloc.lower().split(":")[0] if target_url else ""
        host_parts = [p for p in target_host.split(".") if p]
        root_domain = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else target_host

        # Build set of immune hosts from endpoints with score > 0 or odds_count > 0
        immune_hosts: Set[str] = set()
        for res in classified_results:
            if res.score > 0.0 or res.odds_count > 0:
                host = urllib.parse.urlparse(res.url).netloc.lower().split(":")[0]
                if host:
                    immune_hosts.add(host)

        def is_immune(host: str) -> bool:
            if not host:
                return True
            if host in immune_hosts:
                return True
            if root_domain:
                if host == root_domain or host.endswith("." + root_domain):
                    return True
            if target_host:
                if host == target_host or host.endswith("." + target_host):
                    return True
            return False

        filters_file = pathlib.Path(filters_path)
        existing_ignored: Set[str] = set()
        filters_data: Dict[str, Any] = {}

        if filters_file.exists():
            try:
                with open(filters_file, "r", encoding="utf-8") as f:
                    filters_data = json.load(f)
                existing_ignored = {d.lower() for d in filters_data.get("ignored_domains", [])}
            except Exception as e:
                logging.warning(f"Could not load {filters_path}: {e}")

        domain_counts: Dict[str, int] = {}
        for ev in events:
            host = urllib.parse.urlparse(ev.url).netloc.lower()
            if not host or is_immune(host) or host in existing_ignored:
                continue
            domain_counts[host] = domain_counts.get(host, 0) + 1

        candidate_domains = sorted(domain_counts.keys(), key=lambda d: domain_counts[d], reverse=True)
        if not candidate_domains:
            return

        print("\n=======================================================")
        print("     POST-DISCOVERY NOISE ASSISTANT (TELEMETRY)        ")
        print("=======================================================")
        print("[*] Detected third-party noise/telemetry domains in residual traffic:")
        for domain in candidate_domains:
            print(f"    - {domain} ({domain_counts[domain]} requests)")

        response = input("\nDo you want to append these domains to 'ignored_domains' in network_filters.json? [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            updated_list = filters_data.get("ignored_domains", [])
            for domain in candidate_domains:
                if domain not in updated_list:
                    updated_list.append(domain)
            filters_data["ignored_domains"] = updated_list

            try:
                with open(filters_file, "w", encoding="utf-8") as f:
                    json.dump(filters_data, f, indent=2)
                print(f"[+] Successfully appended {len(candidate_domains)} domains to {filters_path}.")
            except Exception as err:
                print(f"[-] Failed to update {filters_path}: {err}")
        else:
            print("[*] Skipped adding noise domains.")
