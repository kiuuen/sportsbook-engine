import json
import logging
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ReplayExecutor:

    def __init__(self, schema_source: Union[str, pathlib.Path]) -> None:
        self.schema_path: pathlib.Path = self._resolve_schema_path(schema_source)
        self.schema_data: Dict[str, Any] = self._load_schema()
        self.target_domain: str = self.schema_data.get("target_domain", "unknown_target")
        self.session_headers: Dict[str, str] = self.schema_data.get("session", {}).get("essential_headers", {})
        self.session_cookies: Dict[str, str] = self.schema_data.get("session", {}).get("cookies", {})
        self.endpoints: Dict[str, Dict[str, Any]] = self.schema_data.get("endpoints", {})
        self.dependency_graph: Dict[str, Any] = self.schema_data.get("dependency_graph", {})

    @staticmethod
    def _resolve_schema_path(source: Union[str, pathlib.Path]) -> pathlib.Path:
        path = pathlib.Path(source)
        if path.is_file():
            return path

        if not str(source).endswith(".json"):
            candidate = pathlib.Path("./data/schemas") / f"{source}.json"
            if candidate.is_file():
                return candidate

        candidate_default = pathlib.Path("./data/schemas") / path.name
        if candidate_default.is_file():
            return candidate_default

        raise FileNotFoundError(f"Schema file not found for source: {source}")

    def _load_schema(self) -> Dict[str, Any]:
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            raise RuntimeError(f"Failed to parse schema JSON file at {self.schema_path}: {err}") from err

    def _build_request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {k.lower(): v for k, v in self.session_headers.items() if not k.startswith(":")}
        if self.session_cookies:
            cookie_str = "; ".join(f"{k}={v}" for k, v in self.session_cookies.items())
            headers["cookie"] = cookie_str
        return headers

    def hydrate_url(self, template_url: str, params: Optional[Dict[str, Any]] = None) -> str:
        hydrated = template_url
        if params:
            for key, val in params.items():
                placeholder = f"{{{key}}}"
                if placeholder in hydrated:
                    hydrated = hydrated.replace(placeholder, str(val))

        def _fallback_replacer(match: re.Match) -> str:
            param_name = match.group(1).lower()
            if param_name == "uuid":
                return "00000000-0000-0000-0000-000000000000"
            if "timestamp" in param_name or "time" in param_name:
                return "0"
            return "1"

        return re.sub(r"\{([^}]+)\}", _fallback_replacer, hydrated)

    def _harvest_identifiers(self, node: Any, target_params: Set[str], harvested: Dict[str, List[Any]]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                k_lower = str(k).lower()
                for param in target_params:
                    param_clean = param.lower().replace("{", "").replace("}", "")
                    if param_clean in k_lower or k_lower in param_clean or k_lower.endswith("id"):
                        if isinstance(v, (str, int)) and not isinstance(v, bool):
                            harvested.setdefault(param, []).append(v)
                self._harvest_identifiers(v, target_params, harvested)
        elif isinstance(node, list):
            for item in node:
                self._harvest_identifiers(item, target_params, harvested)

    def _resolve_dynamic_params(self, endpoint_key: str, provided_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        endpoint_info = self.endpoints.get(endpoint_key, {})
        required_params = endpoint_info.get("dynamic_parameters", [])
        resolved: Dict[str, Any] = dict(provided_params) if provided_params else {}

        missing_params = [p for p in required_params if p not in resolved]
        if not missing_params:
            return resolved

        # Identify parent producer endpoints (preferring producers mapped in dependency_graph, then catalog_root/fixtures_list)
        parent_candidates: List[str] = []
        producers_map = self.dependency_graph.get("producers", {})
        for param in missing_params:
            if param in producers_map:
                for producer_key in producers_map[param]:
                    if producer_key not in parent_candidates and producer_key != endpoint_key:
                        parent_candidates.append(producer_key)

        for parent_key, parent_info in self.endpoints.items():
            if parent_key == endpoint_key or parent_key in parent_candidates:
                continue
            role = parent_info.get("role", "")
            if role in ("catalog_root", "fixtures_list"):
                parent_candidates.append(parent_key)

        for parent_key in parent_candidates:
            try:
                logging.info(f"[ReplayExecutor] Chaining execution: resolving {missing_params} via parent '{parent_key}'...")
                parent_response = self.execute(parent_key)
                harvested: Dict[str, List[Any]] = {}
                self._harvest_identifiers(parent_response, set(missing_params), harvested)

                for param in list(missing_params):
                    values = harvested.get(param, [])
                    if values:
                        resolved[param] = values[0]
                        missing_params.remove(param)

                if not missing_params:
                    break
            except Exception as err:
                logging.warning(f"[ReplayExecutor] Failed harvesting from parent '{parent_key}': {err}")

        return resolved

    def check_session_health(self, endpoint_key: Optional[str] = None) -> bool:
        if not self.endpoints:
            logging.warning("[ReplayExecutor] No endpoints found in schema.")
            return False

        target_key = endpoint_key or next(iter(self.endpoints.keys()))
        endpoint_info = self.endpoints.get(target_key)
        if not endpoint_info:
            logging.warning(f"[ReplayExecutor] Endpoint key '{target_key}' not found in schema.")
            return False

        resolved_params = self._resolve_dynamic_params(target_key)
        template_url = endpoint_info.get("template_url", "")
        method = endpoint_info.get("method", "GET")

        clean_url = self.hydrate_url(template_url, resolved_params)
        headers = self._build_request_headers()

        try:
            req = urllib.request.Request(url=clean_url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=10.0) as response:
                if response.status == 200:
                    body_text = response.read().decode("utf-8", errors="replace")
                    try:
                        json.loads(body_text)
                        logging.info(f"[ReplayExecutor] Session health probe succeeded (HTTP 200 JSON) on '{target_key}'.")
                        return True
                    except json.JSONDecodeError:
                        logging.warning("[ReplayExecutor] Health probe returned HTTP 200, but payload was not valid JSON.")
                        return False
                else:
                    logging.warning(f"[ReplayExecutor] Health probe returned non-200 status: {response.status}")
                    return False
        except urllib.error.HTTPError as http_err:
            logging.warning(f"[ReplayExecutor] Session health check failed with HTTP status {http_err.code}.")
            return False
        except Exception as err:
            logging.warning(f"[ReplayExecutor] Session health probe error: {err}")
            return False

    def execute(
        self, endpoint_key: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        endpoint_info = self.endpoints.get(endpoint_key)
        if not endpoint_info:
            raise KeyError(f"Endpoint '{endpoint_key}' not present in loaded schema.")

        resolved_params = self._resolve_dynamic_params(endpoint_key, params)
        template_url = endpoint_info.get("template_url", "")
        method = endpoint_info.get("method", "GET")
        hydrated_url = self.hydrate_url(template_url, resolved_params)
        headers = self._build_request_headers()

        req = urllib.request.Request(url=hydrated_url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=12.0) as response:
                status_code = response.status
                body_bytes = response.read()
                body_text = body_bytes.decode("utf-8", errors="replace")

                if status_code != 200:
                    raise RuntimeError(f"HTTP request failed with status code {status_code}")

                return json.loads(body_text)
        except urllib.error.HTTPError as http_err:
            raise RuntimeError(f"HTTP error occurred during execution: {http_err.code} {http_err.reason}") from http_err
        except Exception as err:
            raise RuntimeError(f"Execution error for endpoint '{endpoint_key}': {err}") from err

    def to_curl(self, endpoint_key: str, params: Optional[Dict[str, Any]] = None) -> str:
        endpoint_info = self.endpoints.get(endpoint_key)
        if not endpoint_info:
            raise KeyError(f"Endpoint '{endpoint_key}' not present in loaded schema.")

        resolved_params = self._resolve_dynamic_params(endpoint_key, params)
        template_url = endpoint_info.get("template_url", "")
        method = endpoint_info.get("method", "GET")
        hydrated_url = self.hydrate_url(template_url, resolved_params)
        headers = self._build_request_headers()

        header_flags = " ".join([
            '-H "' + k.lower() + ": " + v.replace('"', '\\"') + '"'
            for k, v in headers.items()
        ])
        if header_flags:
            return f'curl -X {method} "{hydrated_url}" {header_flags}'
        return f'curl -X {method} "{hydrated_url}"'


if __name__ == "__main__":
    schema_dir = pathlib.Path("./data/schemas")
    schema_files = sorted(list(schema_dir.glob("*.json"))) if schema_dir.exists() else []

    if not schema_files:
        print("[-] No schema JSON files found in ./data/schemas/. Please run discovery first.")
    else:
        print("\n=======================================================")
        print("           REPLAY EXECUTOR INTERACTIVE CLI             ")
        print("=======================================================")
        print("[*] Available Target Schemas:")
        for idx, s_file in enumerate(schema_files, start=1):
            print(f"    [{idx}] {s_file.name}")

        selected_file: Optional[pathlib.Path] = None
        while selected_file is None:
            user_input = input(f"\nSelect a schema number (1-{len(schema_files)}): ").strip()
            if user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(schema_files):
                    selected_file = schema_files[choice - 1]
                else:
                    print(f"[-] Invalid selection. Please enter a number between 1 and {len(schema_files)}.")
            else:
                print("[-] Invalid input. Please enter a numeric value.")

        print(f"\n[*] Loading schema blueprint: {selected_file.name}")
        executor = ReplayExecutor(selected_file)

        is_healthy = executor.check_session_health()
        print(f"[+] Session Health Status: {'ACTIVE (200 OK)' if is_healthy else 'EXPIRED / INVALID'}")

        endpoint_keys = list(executor.endpoints.keys())
        if not endpoint_keys:
            print("[-] No endpoints found in selected schema.")
        else:
            print("\n-------------------------------------------------------")
            print("Available Endpoints:")
            for idx, ep_key in enumerate(endpoint_keys, start=1):
                print(f"    [{idx}] {ep_key}")
            batch_option_idx = len(endpoint_keys) + 1
            print(f"    [{batch_option_idx}] [Batch Test] Probe all endpoints in sequence")

            selected_ep_idx: Optional[int] = None
            while selected_ep_idx is None:
                ep_input = input(f"\nSelect an endpoint option (1-{batch_option_idx}): ").strip()
                if ep_input.isdigit():
                    ep_choice = int(ep_input)
                    if 1 <= ep_choice <= batch_option_idx:
                        selected_ep_idx = ep_choice
                    else:
                        print(f"[-] Invalid selection. Enter a number between 1 and {batch_option_idx}.")
                else:
                    print("[-] Invalid input. Please enter a numeric value.")

            if selected_ep_idx == batch_option_idx:
                print("\n=======================================================")
                print("            BATCH ENDPOINT HEALTH PROBE RESULTS        ")
                print("=======================================================")
                for ep_key in endpoint_keys:
                    print(f"\n[*] Probing endpoint: {ep_key}")
                    curl_cmd = executor.to_curl(ep_key)
                    print(f"    cURL: {curl_cmd}")
                    try:
                        resp_payload = executor.execute(ep_key)
                        print("    Status: [SUCCESS 200 OK]")
                        json_str = json.dumps(resp_payload, indent=2)
                        preview = "\n".join(json_str.splitlines()[:5])
                        print(f"    Preview:\n{preview}")
                    except Exception as err:
                        print(f"    Status: [FAILED] - {err}")
            else:
                target_key = endpoint_keys[selected_ep_idx - 1]
                print("\n=======================================================")
                print(f"         SINGLE ENDPOINT EXECUTION: {target_key}")
                print("=======================================================")
                curl_cmd = executor.to_curl(target_key)
                print(f"[*] Generated Escaped cURL:\n    {curl_cmd}\n")
                print(f"[*] Executing live fetch for '{target_key}'...")
                try:
                    response_data = executor.execute(target_key)
                    json_str = json.dumps(response_data, indent=2)
                    preview = "\n".join(json_str.splitlines()[:10])
                    print(f"[+] Response Preview (First 10 lines):\n{preview}")
                except Exception as ex:
                    print(f"[-] Execution failed: {ex}")
