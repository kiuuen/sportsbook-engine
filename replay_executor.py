import json
import logging
import pathlib
import re
import sqlite3
import urllib.parse
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from curl_cffi.requests import Session
from curl_cffi.requests.exceptions import HTTPError, RequestException
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Maximum number of parent-endpoint chaining levels allowed when resolving
# dynamic parameters, preventing circular/infinite recursion between endpoints.
_MAX_CHAIN_DEPTH: int = 3

_IMPERSONATION_TARGET: str = "chrome124"

_PAGINATION_LEXICON_PATH = pathlib.Path(__file__).with_name("sportsbook_lexicon.json")


@lru_cache(maxsize=1)
def _load_pagination_config() -> Tuple[FrozenSet[str], Optional[int], Optional[int]]:
    """Load pagination vocabulary and scaling values from the sportsbook lexicon.

    Returns a tuple of (parameter names, low-value threshold, batch capacity).
    Threshold and capacity are ``None`` when the lexicon does not define them;
    callers must skip upscaling in that case rather than assume defaults.
    """
    try:
        data = json.loads(_PAGINATION_LEXICON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logging.warning(f"[ReplayExecutor] Failed to load pagination lexicon: {err}")
        return frozenset(), None, None

    names = frozenset(
        str(term).strip().lower()
        for term in data.get("pagination_parameters", [])
        if isinstance(term, str) and term.strip()
    )
    threshold: Optional[int] = None
    capacity: Optional[int] = None
    scaling = data.get("pagination_scaling")
    if isinstance(scaling, dict):
        try:
            threshold = int(scaling["low_value_threshold"])
            capacity = int(scaling["batch_capacity"])
        except (KeyError, TypeError, ValueError) as err:
            logging.warning(
                f"[ReplayExecutor] Invalid pagination_scaling in lexicon: {err}"
            )
    return names, threshold, capacity

class ReplayExecutionError(RuntimeError):
    """Raised when a replay request fails for a cause that was already logged."""

class ReplayExecutor:

    @staticmethod
    def _lookup_value(source: Any, *keys: str) -> Any:
        """Return the first non-null value for ``keys`` from a mapping-like object."""
        if source is None:
            return None
        for key in keys:
            value: Any = None
            try:
                if isinstance(source, dict):
                    value = source.get(key)
                else:
                    value = source[key]
            except (KeyError, IndexError, TypeError):
                value = None
            except Exception:
                value = None
            if value is not None:
                return value
        return None

    @staticmethod
    def _coerce_json_value(raw: Any, default: Any) -> Any:
        """Decode JSON text when required, returning ``default`` on failure."""
        if raw is None:
            return default
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                return default
        return raw

    @classmethod
    def _coerce_mapping(cls, raw: Any) -> Dict[str, Any]:
        """Return a dictionary view of a raw mapping or JSON object payload."""
        value = cls._coerce_json_value(raw, {})
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def _coerce_list(cls, raw: Any) -> List[Any]:
        """Return a list view of a raw sequence or JSON array payload."""
        value = cls._coerce_json_value(raw, [])
        return list(value) if isinstance(value, list) else []

    def __init__(
        self,
        schema_source: Union[str, pathlib.Path, int],
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.db_manager = db_manager
        self.target_id: Optional[int] = None
        self.schema_path: Optional[pathlib.Path] = None
        self.schema_data: Dict[str, Any] = {}
        self.target_domain = "unknown_target"
        self.session_headers: Dict[str, str] = {}
        self.session_cookies: Dict[str, str] = {}
        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.dependency_graph: Dict[str, Any] = {}
        self.session: Optional[Session] = None

        # Case 1: an explicit target identifier.
        if isinstance(schema_source, int) and not isinstance(schema_source, bool):
            if not self.db_manager:
                self.db_manager = DatabaseManager(db_path="./data/control_plane.db")
            self._load_from_db(int(schema_source))
            return

        source_path = pathlib.Path(schema_source)

        # Case 2: an existing JSON schema file.
        if source_path.is_file():
            self.schema_path = source_path
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except Exception as err:
                raise RuntimeError(
                    f"Failed to parse schema JSON file at {self.schema_path}: {err}"
                ) from err
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"Schema JSON file at {self.schema_path} must contain a JSON object."
                )
            self.schema_data = loaded
            self._apply_schema_data()
            self._link_to_database_target()
            return

        # Case 3: a domain name or a numeric target identifier.
        if not self.db_manager:
            self.db_manager = DatabaseManager(db_path="./data/control_plane.db")

        domain_source = str(schema_source).strip()
        if domain_source.isdigit():
            self._load_from_db(int(domain_source))
            return

        target = self.db_manager.get_target_by_domain(domain_source)
        if not target:
            raise ValueError(f"Target domain '{domain_source}' not found in database.")
        self._load_from_db(int(target["id"]))

    def _apply_schema_data(self) -> None:
        """Populate executor attributes from the current schema dictionary."""
        self.target_domain = str(self.schema_data.get("target_domain", "unknown_target"))
        session_data = self.schema_data.get("session", {})
        self.session_headers = self._coerce_mapping(
            self._lookup_value(session_data, "essential_headers", "essential_headers_json")
        )
        self.session_cookies = self._coerce_mapping(
            self._lookup_value(session_data, "cookies", "cookies_json")
        )
        endpoints = self.schema_data.get("endpoints", {})
        self.endpoints = endpoints if isinstance(endpoints, dict) else {}
        dependency_graph = self.schema_data.get("dependency_graph", {})
        self.dependency_graph = dependency_graph if isinstance(dependency_graph, dict) else {}
        self.reset_session()

    def _create_session(self) -> Session:
        """Build a curl_cffi session with Chrome impersonation and stored cookies."""
        session = Session(impersonate=_IMPERSONATION_TARGET, verify=False)
        for name, value in self.session_cookies.items():
            if name is None or value is None:
                continue
            try:
                session.cookies.set(str(name), str(value))
            except Exception as err:
                logging.warning(f"[ReplayExecutor] Failed to set cookie '{name}': {err}")
        return session

    def reset_session(self) -> None:
        """Close and rebuild the HTTP session (call after credentials/schema reload)."""
        old_session = getattr(self, "session", None)
        if old_session is not None:
            try:
                old_session.close()
            except Exception as err:
                logging.warning(f"[ReplayExecutor] Failed to close old session: {err}")
        self.session = self._create_session()

    def _link_to_database_target(self) -> None:
        """Associate a file-based schema with its database target when possible."""
        if self.db_manager is None:
            return
        try:
            target = self.db_manager.get_target_by_domain(self.target_domain)
            if target:
                self.target_id = int(target["id"])
        except Exception as err:
            logging.warning(
                f"[ReplayExecutor] Failed to resolve database target for "
                f"'{self.target_domain}': {err}"
            )

    def _load_from_db(self, target_id: int) -> None:
        if not self.db_manager:
            raise RuntimeError("DatabaseManager not provided for DB loading.")

        conn = self.db_manager.get_connection()
        try:
            row = conn.execute("SELECT * FROM targets WHERE id = ?", (target_id,)).fetchone()
        except sqlite3.Error as err:
            raise RuntimeError(
                f"Failed to read target {target_id} from the database: {err}"
            ) from err
        if not row:
            raise ValueError(f"Target with id {target_id} not found.")
        target_row = dict(row)

        target_domain = target_row["domain"]
        provider_family = target_row["provider_family"]
        discovered_at = target_row["discovered_at"]

        # Session data may be exposed as parsed dictionaries or raw JSON text.
        session_row = self.db_manager.get_session_by_target_id(target_id)
        session_headers = self._coerce_mapping(
            self._lookup_value(session_row, "essential_headers", "essential_headers_json")
        )
        session_cookies = self._coerce_mapping(
            self._lookup_value(session_row, "cookies", "cookies_json")
        )

        endpoint_rows = self.db_manager.get_endpoints_by_target_id(target_id)
        endpoints_dict: Dict[str, Dict[str, Any]] = {}
        for ep in endpoint_rows:
            ep_key = str(self._lookup_value(ep, "endpoint_key") or "")
            if not ep_key:
                continue
            endpoints_dict[ep_key] = {
                "role": self._lookup_value(ep, "role") or "catalog_root",
                "volatility": self._lookup_value(ep, "volatility") or "static",
                "protocol": self._lookup_value(ep, "protocol") or "https",
                "method": self._lookup_value(ep, "method") or "GET",
                "template_url": self._lookup_value(ep, "template_url") or "",
                "dynamic_parameters": self._coerce_list(
                    self._lookup_value(ep, "dynamic_parameters", "dynamic_parameters_json")
                ),
                "provides_parameters": self._coerce_list(
                    self._lookup_value(ep, "provides_parameters", "provides_parameters_json")
                ),
                "curl_command": self._lookup_value(ep, "curl_command") or "",
                "handshake_headers": self._coerce_mapping(
                    self._lookup_value(ep, "handshake_headers", "handshake_headers_json")
                ),
                "sample_response": self._coerce_json_value(
                    self._lookup_value(ep, "sample_response", "sample_response_json"), {}
                ),
            }

        dep_row = self.db_manager.get_dependency_graph_by_target_id(target_id)
        producers = self._coerce_mapping(
            self._lookup_value(dep_row, "producers", "producers_json")
        )
        consumers = self._coerce_mapping(
            self._lookup_value(dep_row, "consumers", "consumers_json")
        )

        self.schema_data = {
            "target_domain": target_domain,
            "provider_family": provider_family,
            "discovered_at": discovered_at,
            "session": {
                "essential_headers": session_headers,
                "cookies": session_cookies,
            },
            "endpoints": endpoints_dict,
            "dependency_graph": {"producers": producers, "consumers": consumers},
        }

        self.target_id = target_id
        self._apply_schema_data()

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
        if not self.db_manager or self.target_id is None:
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

    # Standard Chrome desktop user agent used when the captured session
    # headers do not include one, preventing bot-filter 403 responses.
    _DEFAULT_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

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
        """Build outbound request headers from the stored session.

        HTTP/2 pseudo-headers (``:authority``, ``:path``, ``:method``,
        ``:scheme``) are stripped, a default Chrome desktop ``User-Agent`` is
        applied when the session does not provide one, and ``Referer`` /
        ``Origin`` headers are derived from the target domain so third-party
        provider gateways (e.g. ``biahosted.com``) accept replayed requests.
        """
        headers: Dict[str, str] = {}
        for key, value in self.session_headers.items():
            if not isinstance(key, str) or value is None:
                continue
            clean_key = key.strip().lower()
            if not clean_key or clean_key.startswith(":"):
                # Skip HTTP/2 pseudo-headers such as :authority, :path, :method, :scheme.
                continue
            headers[clean_key] = str(value)

        # Guarantee a browser-like User-Agent to avoid gateway 403 rejections.
        if "user-agent" not in headers:
            headers["user-agent"] = self._DEFAULT_USER_AGENT

        # Inject Referer/Origin from the target domain so provider gateways
        # treat the replay as a first-party browser request. Session-provided
        # values take precedence via setdefault.
        domain = str(self.target_domain or "").strip().lower()
        if domain and domain != "unknown_target":
            origin = f"https://{domain}"
            headers.setdefault("referer", f"{origin}/")
            headers.setdefault("origin", origin)
        return headers

    @classmethod
    def _upscale_pagination_params(cls, url: str) -> str:
        """Scale restrictive pagination query parameters up to the batch capacity.

        Pagination key names and the threshold/capacity values come exclusively
        from ``sportsbook_lexicon.json``. Only numeric values at or below the
        threshold are replaced; every other query pair, the path, and any
        remaining ``{placeholder}`` segments are preserved verbatim. When the
        lexicon is unavailable or incomplete, the URL is returned unchanged.
        """
        names, threshold, capacity = _load_pagination_config()
        if not names or threshold is None or capacity is None:
            return url

        split = urllib.parse.urlsplit(url)
        if not split.query:
            return url

        adjusted: List[str] = []
        changed = False
        for pair in split.query.split("&"):
            if "=" in pair:
                raw_key, raw_value = pair.split("=", 1)
            else:
                raw_key, raw_value = pair, ""
            key = urllib.parse.unquote_plus(raw_key).strip().lower()
            if (
                key in names
                and raw_value.isdigit()
                and int(raw_value) <= threshold
            ):
                logging.info(
                    f"[ReplayExecutor] Upscaled pagination parameter '{key}' "
                    f"from {raw_value} to {capacity}."
                )
                adjusted.append(f"{raw_key}={capacity}")
                changed = True
            else:
                adjusted.append(pair)

        if not changed:
            return url
        return urllib.parse.urlunsplit(split._replace(query="&".join(adjusted)))

    def hydrate_url(self, template_url: str, params: Optional[Dict[str, Any]] = None) -> str:
        hydrated = template_url
        if params:
            for key, val in params.items():
                placeholder = f"{{{key}}}"
                if placeholder in hydrated:
                    hydrated = hydrated.replace(placeholder, str(val))

        hydrated = self._upscale_pagination_params(hydrated)

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

    def _resolve_dynamic_params(
        self,
        endpoint_key: str,
        provided_params: Optional[Dict[str, Any]] = None,
        visited: Optional[Set[str]] = None,
        depth: int = 0,
    ) -> Dict[str, Any]:
        visited = visited if visited is not None else set()
        visited.add(endpoint_key)
        endpoint_info = self.endpoints.get(endpoint_key, {})
        required_params = endpoint_info.get("dynamic_parameters", [])
        resolved: Dict[str, Any] = dict(provided_params) if provided_params else {}

        missing_params = [p for p in required_params if p not in resolved]
        if not missing_params:
            return resolved

        # Stop chaining when the maximum depth has been reached.
        if depth >= _MAX_CHAIN_DEPTH:
            logging.warning(
                f"[ReplayExecutor] Max chain depth {_MAX_CHAIN_DEPTH} reached at "
                f"'{endpoint_key}'; returning resolved params as-is."
            )
            return resolved

        # Identify parent producer endpoints (preferring producers mapped in
        # dependency_graph, then catalog_root/fixtures_list), skipping any
        # endpoint already visited in the current chain to prevent cycles.
        parent_candidates: List[str] = []
        producers_map = self.dependency_graph.get("producers", {})
        for param in missing_params:
            if param in producers_map:
                for producer_key in producers_map[param]:
                    if (
                        producer_key not in parent_candidates
                        and producer_key != endpoint_key
                        and producer_key not in visited
                    ):
                        parent_candidates.append(producer_key)

        for parent_key, parent_info in self.endpoints.items():
            if (
                parent_key == endpoint_key
                or parent_key in parent_candidates
                or parent_key in visited
            ):
                continue
            role = parent_info.get("role", "")
            if role in ("catalog_root", "fixtures_list"):
                parent_candidates.append(parent_key)

        for parent_key in parent_candidates:
            if parent_key in visited:
                continue
            visited.add(parent_key)
            try:
                logging.info(
                    f"[ReplayExecutor] Chaining execution: resolving {missing_params} "
                    f"via parent '{parent_key}' (depth {depth + 1})..."
                )
                parent_response = self.execute(
                    parent_key, visited=visited, depth=depth + 1
                )
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
                logging.warning(
                    f"[ReplayExecutor] Failed harvesting from parent '{parent_key}': {err}"
                )

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
        if self.session is None:
            self.reset_session()

        try:
            response = self.session.request(
                method,
                clean_url,
                headers=headers,
                timeout=10.0,
                allow_redirects=True,
            )
            status_code = int(response.status_code)
            if status_code == 200:
                body_text = response.text
                try:
                    json.loads(body_text)
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
                self._log_health(
                    endpoint_key=target_key,
                    status="FAILED",
                    status_code=status_code,
                    message=f"Health probe returned non-200 status: {status_code}",
                    response_preview="",
                )
                logging.warning(f"[ReplayExecutor] Health probe returned non-200 status: {status_code}")
                return False
        except HTTPError as http_err:
            status_code = int(getattr(getattr(http_err, "response", None), "status_code", -1) or -1)
            self._log_health(
                endpoint_key=target_key,
                status="FAILED",
                status_code=status_code,
                message=f"Session health check failed with HTTP status {status_code}: {http_err}",
                response_preview="",
            )
            logging.warning(f"[ReplayExecutor] Session health check failed with HTTP status {status_code}.")
            return False
        except RequestException as req_err:
            self._log_health(
                endpoint_key=target_key,
                status="FAILED",
                status_code=-1,
                message=f"Session health probe transport error: {req_err}",
                response_preview="",
            )
            logging.warning(f"[ReplayExecutor] Session health probe transport error: {req_err}")
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
        self,
        endpoint_key: str,
        params: Optional[Dict[str, Any]] = None,
        visited: Optional[Set[str]] = None,
        depth: int = 0,
    ) -> Any:
        endpoint_info = self.endpoints.get(endpoint_key)
        if not endpoint_info:
            raise KeyError(f"Endpoint '{endpoint_key}' not present in loaded schema.")

        resolved_params = self._resolve_dynamic_params(
            endpoint_key, params, visited=visited, depth=depth
        )
        template_url = endpoint_info.get("template_url", "")
        method = endpoint_info.get("method", "GET")
        hydrated_url = self.hydrate_url(template_url, resolved_params)
        headers = self._build_request_headers()
        if self.session is None:
            self.reset_session()

        try:
            response = self.session.request(
                method,
                hydrated_url,
                headers=headers,
                timeout=12.0,
                allow_redirects=True,
            )
            status_code = int(response.status_code)
            body_text = response.text

            if status_code != 200:
                self._log_health(
                    endpoint_key=endpoint_key,
                    status="FAILED",
                    status_code=status_code,
                    message=f"HTTP request failed with status code {status_code}",
                    response_preview="",
                )
                raise ReplayExecutionError(f"HTTP request failed with status code {status_code}")

            try:
                payload = json.loads(body_text)
            except json.JSONDecodeError as jde:
                self._log_health(
                    endpoint_key=endpoint_key,
                    status="FAILED",
                    status_code=status_code,
                    message=f"Response payload was not valid JSON: {jde}",
                    response_preview=self.clamp_preview(body_text, max_lines=5, max_chars=500),
                )
                raise RuntimeError("Response payload was not valid JSON") from jde

            self._log_health(
                endpoint_key=endpoint_key,
                status="SUCCESS",
                status_code=200,
                message="Replay execution succeeded (HTTP 200 JSON).",
                response_preview=self.clamp_preview(payload, max_lines=5, max_chars=500),
            )
            return payload

        except ReplayExecutionError:
            raise
        except HTTPError as http_err:
            status_code = int(getattr(getattr(http_err, "response", None), "status_code", -1) or -1)
            self._log_health(
                endpoint_key=endpoint_key,
                status="FAILED",
                status_code=status_code,
                message=f"HTTP error during execution: {http_err}",
                response_preview="",
            )
            raise ReplayExecutionError(f"HTTP request failed with status code {status_code}: {http_err}") from http_err
        except RequestException as req_err:
            self._log_health(
                endpoint_key=endpoint_key,
                status="FAILED",
                status_code=-1,
                message=f"Transport error during execution: {req_err}",
                response_preview="",
            )
            raise RuntimeError(f"Transport error for endpoint '{endpoint_key}': {req_err}") from req_err
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
    def _format_error(err: BaseException) -> str:
        """Flatten an exception and its cause chain into one readable message."""
        parts: List[str] = [str(err)]
        cause: Optional[BaseException] = err.__cause__
        while cause is not None:
            parts.append(str(cause))
            cause = cause.__cause__
        return " -> ".join(part for part in parts if part)

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
