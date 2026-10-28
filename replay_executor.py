import json
import logging
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Union

from database_manager import DatabaseManager
from heuristic_classifier import AutomatedReplayProber, HeuristicClassifier, RouteTemplateDeducer, NoiseAssistant
from browser_orchestrator import ChromeOrchestrator
from cdp_interceptor import NetworkEvent, CDPNetworkInterceptor
from schema_exporter import SchemaExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ReplayExecutor:

    def __init__(
        self,
        schema_source: Union[str, pathlib.Path, int],
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        # Store db_manager for later health logging; create default if none provided
        self.db_manager = db_manager
        if isinstance(schema_source, pathlib.Path):
            source_path = schema_source
        else:
            source_path = pathlib.Path(schema_source)

        # Case 1: existing JSON file
        if source_path.is_file():
            self.schema_path = source_path
            # Original JSON loading logic (unchanged)
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    self.schema_data = json.load(f)
            except Exception as err:
                raise RuntimeError(f"Failed to parse schema JSON file at {self.schema_path}: {err}") from err

            # Derive attributes used later
            self.target_domain = self.schema_data.get("target_domain", "unknown_target")
            self.session_headers = self.schema_data.get("session", {}).get("essential_headers", {})
            self.session_cookies = self.schema_data.get("session", {}).get("cookies", {})
            self.endpoints = self.schema_data.get("endpoints", {})
            self.dependency_graph = self.schema_data.get("dependency_graph", {})
            return

        # Cases 2 & 3: domain string or target_id integer – require DatabaseManager
        if not self.db_manager:
            self.db_manager = DatabaseManager(db_path="./data/control_plane.db")

        # Try interpreting source as an integer target_id
        try:
            target_id = int(schema_source)
            self._load_from_db(target_id)
            return
        except ValueError:
            # Not an integer – treat as domain string
            pass

        # Domain lookup
        target = self.db_manager.get_target_by_domain(schema_source)
        if not target:
            raise ValueError(f"Target domain '{schema_source}' not found in database.")
        self._load_from_db(target["id"])

    def _load_from_db(self, target_id: int) -> None:
        """
        Populate the ReplayExecutor from the SQLite control plane using the given target_id.
        Raises ValueError if the target does not exist.
        """
        if not self.db_manager:
            raise RuntimeError("DatabaseManager not provided for DB loading.")

        # Fetch target record
        conn = self.db_manager.get_connection()
        row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        if not row:
            raise ValueError(f"Target with id {target_id} not found.")
        target_row = dict(row)

        target_domain = target_row["domain"]
        provider_family = target_row["provider_family"]
        discovered_at = target_row["discovered_at"]
        # canonical_pipeline_json is not needed for ReplayExecutor

        # Fetch session
        session_row = self.db_manager.get_session_by_target_id(target_id)
        session_headers: Dict[str, str] = {}
        session_cookies: Dict[str, str] = {}
        if session_row:
            raw_headers = session_row.get("essential_headers")
            if raw_headers is None:
                raw_headers = session_row.get("essential_headers_json", {})
            if isinstance(raw_headers, str):
                session_headers = json.loads(raw_headers)
            elif isinstance(raw_headers, dict):
                session_headers = raw_headers

            raw_cookies = session_row.get("cookies")
            if raw_cookies is None:
                raw_cookies = session_row.get("cookies_json", {})
            if isinstance(raw_cookies, str):
                session_cookies = json.loads(raw_cookies)
            elif isinstance(raw_cookies, dict):
                session_cookies = raw_cookies

        # Fetch endpoints
        endpoint_rows = self.db_manager.get_endpoints_by_target_id(target_id)
        endpoints_dict: Dict[str, Dict[str, Any]] = {}
        for ep in endpoint_rows:
            ep_key = ep["endpoint_key"]
            endpoint_info = {
                "role": ep["role"],
                "volatility": ep["volatility"],
                "protocol": ep["protocol"],
                "method": ep["method"],
                "template_url": ep["template_url"],
                "dynamic_parameters": json.loads(ep["dynamic_parameters_json"]),
                "provides_parameters": json.loads(ep["provides_parameters_json"]),
                "curl_command": ep["curl_command"],
                "handshake_headers": json.loads(ep["handshake_headers_json"]),
                "sample_response": json.loads(ep["sample_response_json"]),
            }
            # If duplicate keys exist, the last one wins – acceptable for this use-case
            endpoints_dict[ep_key] = endpoint_info

        # Fetch dependency graph
        dep_row = self.db_manager.get_dependency_graph_by_target_id(target_id)
        producers = json.loads(dep_row["producers_json"]) if dep_row else {}
        consumers = json.loads(dep_row["consumers_json"]) if dep_row else {}
        dependency_graph = {"producers": producers, "consumers": consumers}

        # Build schema_data dict matching the structure used by DatabaseManager.save_schema_dict
        self.schema_data = {
            "target_domain": target_domain,
            "provider_family": provider_family,
            "discovered_at": discovered_at,
            "session": {
                "essential_headers": session_headers,
                "cookies": session_cookies,
            },
            "endpoints": endpoints_dict,
            "dependency_graph": dependency_graph,
        }

        # Store target_id for health-logging
        self.target_id = target_id

        # Initialise other attributes used by the original implementation
        self.target_domain = target_domain
        self.session_headers = session_headers
        self.session_cookies = session_cookies
        self.endpoints = endpoints_dict
        self.dependency_graph = dependency_graph

    def _log_health(
        self,
        endpoint_key: str,
        status: str,
        status_code: int,
        message: str,
        response_preview: str = "",
    ) -> None:
        """
        Record a health/probe outcome in replay_health_logs if a DatabaseManager
        and target_id are available. Errors are logged but not propagated.
        """
        if not self.db_manager or not hasattr(self, "target_id"):
            return
        try:
            self.db_manager.log_replay_health(
                target_id=self.target_id,
                endpoint_key=endpoint_key,
                status=status,
                status_code=status_code,
                message=message,
                response_preview=response_preview,
            )
        except Exception as e:
            logging.warning(f"[ReplayExecutor] Failed to log health record: {e}")

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
                status_code = response.status
                if status_code == 200:
                    body_text = response.read().decode("utf-8", errors="replace")
                    try:
                        json.loads(body_text)
                        # Successful probe – log to DB
                        self._log_health(
                            endpoint_key=target_key,
                            status="SUCCESS",
                            status_code=200,
                            message="Session health probe succeeded (HTTP 200 JSON).",
                            response_preview=self.clamp_preview(body_text, max_lines=5, max_chars=500),
                        )
                        logging.info(f"[ReplayExecutor] Session health probe succeeded (HTTP 200 JSON) on '{target_key}'.")
                        return True
                    except json.JSONDecodeError:
                        # 200 but not JSON
                        self._log_health(
                            endpoint_key=target_key,
                            status="FAILED",
                            status_code=200,
                            message="Health probe returned HTTP 200, but payload was not valid JSON.",
                            response_preview="",
                        )
                        logging.warning("[ReplayExecutor] Health probe returned HTTP 200, but payload was not valid JSON.")
                        return False
                else:
                    # Non-200 response
                    self._log_health(
                        endpoint_key=target_key,
                        status="FAILED",
                        status_code=status_code,
                        message=f"Health probe returned non-200 status: {status_code}",
                        response_preview="",
                    )
                    logging.warning(f"[ReplayExecutor] Health probe returned non-200 status: {status_code}")
                    return False
        except urllib.error.HTTPError as http_err:
            self._log_health(
                endpoint_key=target_key,
                status="FAILED",
                status_code=http_err.code,
                message=f"Session health check failed with HTTP status {http_err.code}: {http_err.reason}",
                response_preview="",
            )
            logging.warning(f"[ReplayExecutor] Session health check failed with HTTP status {http_err.code}.")
            return False
        except Exception as err:
            self._log_health(
                endpoint_key=target_key,
                status="FAILED",
                status_code=-1,
                message=f"Session health probe error: {err}",
                response_preview="",
            )
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
                    # Log failure before raising
                    self._log_health(
                        endpoint_key=endpoint_key,
                        status="FAILED",
                        status_code=status_code,
                        message=f"HTTP request failed with status code {status_code}",
                        response_preview="",
                    )
                    raise RuntimeError(f"HTTP request failed with status code {status_code}")

                try:
                    payload = json.loads(body_text)
                except json.JSONDecodeError as jde:
                    # Log non-JSON response failure
                    self._log_health(
                        endpoint_key=endpoint_key,
                        status="FAILED",
                        status_code=status_code,
                        message=f"Response payload was not valid JSON: {jde}",
                        response_preview=self.clamp_preview(body_text, max_lines=5, max_chars=500),
                    )
                    raise RuntimeError("Response payload was not valid JSON") from jde

                # Success – log and return
                self._log_health(
                    endpoint_key=endpoint_key,
                    status="SUCCESS",
                    status_code=200,
                    message="Replay execution succeeded (HTTP 200 JSON).",
                    response_preview=self.clamp_preview(payload, max_lines=5, max_chars=500),
                )
                return payload

        except urllib.error.HTTPError as http_err:
            self._log_health(
                endpoint_key=endpoint_key,
                status="FAILED",
                status_code=http_err.code,
                message=f"HTTP error during execution: {http_err.code} {http_err.reason}",
                response_preview="",
            )
            raise RuntimeError(
                f"HTTP error occurred during execution: {http_err.code} {http_err.reason}"
            ) from http_err

        except Exception as err:
            self._log_health(
                endpoint_key=endpoint_key,
                status="FAILED",
                status_code=-1,
                message=f"Execution error: {err}",
                response_preview="",
            )
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

    @staticmethod
    def clamp_preview(data: Any, max_lines: int = 10, max_chars: int = 1000) -> str:
        if isinstance(data, (dict, list)):
            text = json.dumps(data, indent=2)
        else:
            text = str(data)

        lines = text.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True

        clamped_lines: List[str] = []
        for line in lines:
            if len(line) > 200:
                clamped_lines.append(line[:200] + "...")
                truncated = True
            else:
                clamped_lines.append(line)

        result = "\n".join(clamped_lines)
        if len(result) > max_chars:
            result = result[:max_chars] + "... [truncated]"
        elif truncated:
            result += "\n... [truncated]"

        return result

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
                        preview = ReplayExecutor.clamp_preview(resp_payload, max_lines=5, max_chars=500)
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
                    preview = ReplayExecutor.clamp_preview(response_data, max_lines=10, max_chars=1000)
                    print(f"[+] Response Preview:\n{preview}")
                except Exception as ex:
                    print(f"[-] Execution failed: {ex}")
