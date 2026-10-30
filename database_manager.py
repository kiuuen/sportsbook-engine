import datetime
import json
import logging
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class DatabaseManager:
    """Control Plane persistence layer managing targets, sessions, endpoints,
    dependency graphs, and replay logs in SQLite with ANSI-SQL compatibility.
    """

    def __init__(self, db_path: Union[str, pathlib.Path] = "./data/control_plane.db") -> None:
        self.db_path: pathlib.Path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as err:
                logging.warning(f"[DB] Error closing database connection: {err}")
            finally:
                self._conn = None

    def __enter__(self) -> "DatabaseManager":
        self.get_connection()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _safe_json_loads(raw: Any, default: Any) -> Any:
        """Decode a stored JSON payload without raising on malformed data."""
        if raw is None:
            return default
        if isinstance(raw, (dict, list)):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as err:
                logging.warning(f"[DB] Failed to decode stored JSON payload: {err}")
                return default
        return default

    def _init_db(self) -> None:
        conn = self.get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE NOT NULL,
                    provider_family TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    canonical_pipeline_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL UNIQUE,
                    essential_headers_json TEXT NOT NULL,
                    cookies_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS endpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    endpoint_key TEXT NOT NULL,
                    category TEXT NOT NULL,
                    role TEXT NOT NULL,
                    volatility TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    method TEXT NOT NULL,
                    template_url TEXT NOT NULL,
                    dynamic_parameters_json TEXT NOT NULL,
                    provides_parameters_json TEXT NOT NULL,
                    curl_command TEXT NOT NULL,
                    handshake_headers_json TEXT NOT NULL,
                    sample_response_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(target_id, endpoint_key, category),
                    FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS dependency_graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL UNIQUE,
                    producers_json TEXT NOT NULL,
                    consumers_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE CASCADE
                );
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_health_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    endpoint_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    response_preview TEXT NOT NULL,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES targets(id) ON DELETE CASCADE
                );
            """)

    def upsert_target(
        self,
        domain: str,
        provider_family: str,
        discovered_at: str,
        canonical_pipeline: Dict[str, Any],
    ) -> int:
        conn = self.get_connection()
        now = self._utc_now_iso()
        pipeline_json = json.dumps(canonical_pipeline)

        query = """
            INSERT INTO targets (domain, provider_family, discovered_at, canonical_pipeline_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                provider_family = EXCLUDED.provider_family,
                discovered_at = EXCLUDED.discovered_at,
                canonical_pipeline_json = EXCLUDED.canonical_pipeline_json,
                updated_at = EXCLUDED.updated_at;
        """
        with conn:
            conn.execute(query, (domain, provider_family, discovered_at, pipeline_json, now, now))
            row = conn.execute("SELECT id FROM targets WHERE domain = ?", (domain,)).fetchone()
            if not row:
                raise RuntimeError(f"Failed to retrieve target ID for domain: {domain}")
            return int(row["id"])

    def upsert_session(
        self,
        target_id: int,
        essential_headers: Dict[str, str],
        cookies: Dict[str, str],
    ) -> int:
        conn = self.get_connection()
        now = self._utc_now_iso()
        headers_json = json.dumps(essential_headers)
        cookies_json = json.dumps(cookies)

        query = """
            INSERT INTO sessions (target_id, essential_headers_json, cookies_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(target_id) DO UPDATE SET
                essential_headers_json = EXCLUDED.essential_headers_json,
                cookies_json = EXCLUDED.cookies_json,
                updated_at = EXCLUDED.updated_at;
        """
        with conn:
            conn.execute(query, (target_id, headers_json, cookies_json, now))
            row = conn.execute("SELECT id FROM sessions WHERE target_id = ?", (target_id,)).fetchone()
            if not row:
                raise RuntimeError(f"Failed to retrieve session ID for target_id: {target_id}")
            return int(row["id"])

    def upsert_dependency_graph(
        self,
        target_id: int,
        producers: Dict[str, List[str]],
        consumers: Dict[str, List[str]],
    ) -> int:
        conn = self.get_connection()
        now = self._utc_now_iso()
        producers_json = json.dumps(producers)
        consumers_json = json.dumps(consumers)

        query = """
            INSERT INTO dependency_graphs (target_id, producers_json, consumers_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(target_id) DO UPDATE SET
                producers_json = EXCLUDED.producers_json,
                consumers_json = EXCLUDED.consumers_json,
                updated_at = EXCLUDED.updated_at;
        """
        with conn:
            conn.execute(query, (target_id, producers_json, consumers_json, now))
            row = conn.execute("SELECT id FROM dependency_graphs WHERE target_id = ?", (target_id,)).fetchone()
            if not row:
                raise RuntimeError(f"Failed to retrieve dependency graph ID for target_id: {target_id}")
            return int(row["id"])

    def sync_endpoints(
        self,
        target_id: int,
        endpoints: Dict[str, Dict[str, Any]],
        category: str = "core",
    ) -> None:
        conn = self.get_connection()
        now = self._utc_now_iso()

        query = """
            INSERT INTO endpoints (
                target_id, endpoint_key, category, role, volatility, protocol,
                method, template_url, dynamic_parameters_json, provides_parameters_json,
                curl_command, handshake_headers_json, sample_response_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_id, endpoint_key, category) DO UPDATE SET
                role = EXCLUDED.role,
                volatility = EXCLUDED.volatility,
                protocol = EXCLUDED.protocol,
                method = EXCLUDED.method,
                template_url = EXCLUDED.template_url,
                dynamic_parameters_json = EXCLUDED.dynamic_parameters_json,
                provides_parameters_json = EXCLUDED.provides_parameters_json,
                curl_command = EXCLUDED.curl_command,
                handshake_headers_json = EXCLUDED.handshake_headers_json,
                sample_response_json = EXCLUDED.sample_response_json,
                updated_at = EXCLUDED.updated_at;
        """

        records = []
        for ep_key, ep_val in endpoints.items():
            records.append((
                target_id,
                ep_key,
                category,
                ep_val.get("role", "catalog_root"),
                ep_val.get("volatility", "static"),
                ep_val.get("protocol", "https"),
                ep_val.get("method", "GET"),
                ep_val.get("template_url", ""),
                json.dumps(ep_val.get("dynamic_parameters", [])),
                json.dumps(ep_val.get("provides_parameters", [])),
                ep_val.get("curl_command", ""),
                json.dumps(ep_val.get("handshake_headers", {})),
                json.dumps(ep_val.get("sample_response", {})),
                now,
            ))

        with conn:
            conn.executemany(query, records)

    def save_schema_dict(self, schema_data: Dict[str, Any]) -> int:
        domain = schema_data.get("target_domain", "unknown_target")
        provider_family = schema_data.get("provider_family", "unknown_provider")
        discovered_at = schema_data.get("discovered_at", self._utc_now_iso())
        canonical_pipeline = schema_data.get("canonical_pipeline", {})

        target_id = self.upsert_target(domain, provider_family, discovered_at, canonical_pipeline)

        session_data = schema_data.get("session", {})
        self.upsert_session(
            target_id,
            essential_headers=session_data.get("essential_headers", {}),
            cookies=session_data.get("cookies", {}),
        )

        dep_graph = schema_data.get("dependency_graph", {})
        self.upsert_dependency_graph(
            target_id,
            producers=dep_graph.get("producers", {}),
            consumers=dep_graph.get("consumers", {}),
        )

        self.sync_endpoints(target_id, schema_data.get("endpoints", {}), category="core")
        self.sync_endpoints(target_id, schema_data.get("streaming_endpoints", {}), category="streaming")
        self.sync_endpoints(target_id, schema_data.get("auxiliary_endpoints", {}), category="auxiliary")

        return target_id

    def get_target_by_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        row = conn.execute(
            "SELECT * FROM targets WHERE LOWER(domain) = LOWER(?)", (domain,)
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        res["canonical_pipeline"] = self._safe_json_loads(res.get("canonical_pipeline_json"), {})
        return res

    def get_session_by_target_id(self, target_id: int) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        row = conn.execute("SELECT * FROM sessions WHERE target_id = ?", (target_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "target_id": row["target_id"],
            "essential_headers": self._safe_json_loads(row["essential_headers_json"], {}),
            "cookies": self._safe_json_loads(row["cookies_json"], {}),
            "updated_at": row["updated_at"],
        }

    def get_endpoints_by_target_id(
        self, target_id: int, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        if category:
            rows = conn.execute(
                "SELECT * FROM endpoints WHERE target_id = ? AND category = ?",
                (target_id, category),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM endpoints WHERE target_id = ?", (target_id,)
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            results.append({
                "id": r["id"],
                "target_id": r["target_id"],
                "endpoint_key": r["endpoint_key"],
                "category": r["category"],
                "role": r["role"],
                "volatility": r["volatility"],
                "protocol": r["protocol"],
                "method": r["method"],
                "template_url": r["template_url"],
                "dynamic_parameters": self._safe_json_loads(r["dynamic_parameters_json"], []),
                "provides_parameters": self._safe_json_loads(r["provides_parameters_json"], []),
                "curl_command": r["curl_command"],
                "handshake_headers": self._safe_json_loads(r["handshake_headers_json"], {}),
                "sample_response": self._safe_json_loads(r["sample_response_json"], {}),
                "updated_at": r["updated_at"],
            })
        return results

    def get_dependency_graph_by_target_id(self, target_id: int) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        row = conn.execute("SELECT * FROM dependency_graphs WHERE target_id = ?", (target_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "target_id": row["target_id"],
            "producers": self._safe_json_loads(row["producers_json"], {}),
            "consumers": self._safe_json_loads(row["consumers_json"], {}),
            "updated_at": row["updated_at"],
        }

    def log_replay_health(
        self,
        target_id: int,
        endpoint_key: str,
        status: str,
        status_code: int,
        message: str,
        response_preview: str = "",
    ) -> int:
        conn = self.get_connection()
        now = self._utc_now_iso()
        query = """
            INSERT INTO replay_health_logs (
                target_id, endpoint_key, status, status_code, message, response_preview, executed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with conn:
            cur = conn.execute(
                query,
                (target_id, endpoint_key, status, status_code, message, response_preview, now),
            )
            return cur.lastrowid

    def delete_target(self, domain: str) -> bool:
        """Delete a target and every record associated with it.

        Child rows are removed explicitly so the operation is correct even if
        SQLite foreign-key enforcement is unavailable, in addition to the
        ``ON DELETE CASCADE`` clauses declared on the schema.

        Args:
            domain: Domain of the target to remove.

        Returns:
            True when a target was deleted, False when the domain is unknown.
        """
        if not isinstance(domain, str):
            return False
        normalized = domain.strip().lower()
        if not normalized:
            return False

        conn = self.get_connection()
        try:
            with conn:
                row = conn.execute(
                    "SELECT id FROM targets WHERE LOWER(domain) = ?", (normalized,)
                ).fetchone()
                if row is None:
                    return False
                target_id = int(row["id"])
                conn.execute("DELETE FROM replay_health_logs WHERE target_id = ?", (target_id,))
                conn.execute("DELETE FROM dependency_graphs WHERE target_id = ?", (target_id,))
                conn.execute("DELETE FROM endpoints WHERE target_id = ?", (target_id,))
                conn.execute("DELETE FROM sessions WHERE target_id = ?", (target_id,))
                conn.execute("DELETE FROM targets WHERE id = ?", (target_id,))
            logging.info(
                f"[DB] Deleted target '{normalized}' (id={target_id}) and all associated records."
            )
            return True
        except sqlite3.Error as err:
            logging.error(f"[DB] Failed to delete target '{normalized}': {err}")
            return False

    def get_recent_replay_logs(
        self, target_id: int, limit: int = 50
    ) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT * FROM replay_health_logs
            WHERE target_id = ?
            ORDER BY executed_at DESC
            LIMIT ?
            """,
            (target_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

if __name__ == "__main__":
    test_db_file = pathlib.Path("./data/test_control_plane.db")
    if test_db_file.exists():
        test_db_file.unlink()

    print("[*] Initializing DatabaseManager verification harness...")
    with DatabaseManager(db_path=test_db_file) as db:
        dummy_schema = {
            "target_domain": "sportsbook_example.com",
            "provider_family": "provider_test_1234",
            "discovered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session": {
                "essential_headers": {"user-agent": "Mozilla/5.0", "accept": "application/json"},
                "cookies": {"session_token": "abc123xyz"},
            },
            "canonical_pipeline": {
                "sports_catalog": "GET_https://api.sportsbook_example.com/v1/sports",
                "live_overview": "GET_https://api.sportsbook_example.com/v1/live",
                "upcoming_fixtures": None,
                "match_detail_metrics": None,
            },
            "dependency_graph": {
                "producers": {"sport_id": ["GET_https://api.sportsbook_example.com/v1/sports"]},
                "consumers": {"GET_https://api.sportsbook_example.com/v1/live": ["sport_id"]},
            },
            "endpoints": {
                "GET_https://api.sportsbook_example.com/v1/sports": {
                    "role": "catalog_root",
                    "volatility": "static",
                    "protocol": "https",
                    "method": "GET",
                    "template_url": "https://api.sportsbook_example.com/v1/sports",
                    "dynamic_parameters": [],
                    "provides_parameters": ["sport_id"],
                    "curl_command": 'curl -X GET "https://api.sportsbook_example.com/v1/sports"',
                    "handshake_headers": {},
                    "sample_response": {"sports": [{"id": "1", "name": "Soccer"}]},
                }
            },
            "streaming_endpoints": {
                "WS_wss://stream.sportsbook_example.com/v1/odds": {
                    "role": "odds_detail",
                    "volatility": "realtime",
                    "protocol": "websocket",
                    "method": "WS_RECV",
                    "template_url": "wss://stream.sportsbook_example.com/v1/odds",
                    "dynamic_parameters": ["event_id"],
                    "provides_parameters": [],
                    "curl_command": "",
                    "handshake_headers": {"authorization": "Bearer token123"},
                    "sample_response": {"event_id": "100", "odds": 1.95},
                }
            },
            "auxiliary_endpoints": {},
        }

        print("[*] Performing dummy schema persistence...")
        target_id = db.save_schema_dict(dummy_schema)
        print(f"[+] Schema successfully saved with target_id: {target_id}")

        print("[*] Retrieving persisted target...")
        target_record = db.get_target_by_domain("sportsbook_example.com")
        assert target_record is not None, "Target record should exist"
        print(f"[+] Retrieved Target Domain: {target_record['domain']}, Provider: {target_record['provider_family']}")

        print("[*] Retrieving persisted session...")
        session_record = db.get_session_by_target_id(target_id)
        assert session_record is not None, "Session record should exist"
        print(f"[+] Retrieved Session Headers: {list(session_record['essential_headers'].keys())}")

        print("[*] Retrieving core and streaming endpoints...")
        core_endpoints = db.get_endpoints_by_target_id(target_id, category="core")
        streaming_endpoints = db.get_endpoints_by_target_id(target_id, category="streaming")
        print(f"[+] Core Endpoints count: {len(core_endpoints)}, Streaming Endpoints count: {len(streaming_endpoints)}")

        print("[*] Logging replay health test outcome...")
        log_id = db.log_replay_health(
            target_id=target_id,
            endpoint_key="GET_https://api.sportsbook_example.com/v1/sports",
            status="SUCCESS",
            status_code=200,
            message="Replay probe succeeded",
            response_preview='{"sports": [{"id": "1", "name": "Soccer"}]}',
        )
        print(f"[+] Logged replay outcome with log_id: {log_id}")

        logs = db.get_recent_replay_logs(target_id, limit=10)
        assert len(logs) == 1, "Should have 1 replay log"
        print(f"[+] Verified Replay Log Message: {logs[0]['message']}")

    if test_db_file.exists():
        test_db_file.unlink()
    print("[+] DatabaseManager verification test completed successfully!")
