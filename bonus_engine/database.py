"""SQLite persistence layer for bonus_engine promotional domain models.

This store is fully isolated from the control-plane database: it owns its
own SQLite file, schema, and connection lifecycle. It persists the
relationship between target domains, raw promotional sources (with content
versioning and change detection), and extracted campaigns.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from bonus_engine.models import Promotion, SourceProvenance, VerificationState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BonusDatabaseManager:
    """Dedicated persistence layer for promotions, sources, and targets."""

    def __init__(self, db_path: Union[str, pathlib.Path] = "./data/bonus_engine.db") -> None:
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
                logging.warning(f"[BONUS_DB] Error closing database connection: {err}")
            finally:
                self._conn = None

    def __enter__(self) -> "BonusDatabaseManager":
        self.get_connection()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def _decode_raw_content(raw_content: Union[str, bytes]) -> str:
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, bytes):
            return raw_content.decode("utf-8", errors="replace")
        raise TypeError("raw_content must be a string or bytes")

    def _init_db(self) -> None:
        conn = self.get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bonus_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS promotion_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    source_url TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    extraction_method TEXT NOT NULL DEFAULT 'automated',
                    UNIQUE(target_id, source_url, version),
                    FOREIGN KEY (target_id) REFERENCES bonus_targets(id) ON DELETE CASCADE
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS promotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    promotion_id TEXT UNIQUE NOT NULL,
                    promotion_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    verification_state TEXT NOT NULL,
                    source_id INTEGER,
                    source_url TEXT,
                    extraction_method TEXT NOT NULL DEFAULT 'automated',
                    promotion_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (target_id) REFERENCES bonus_targets(id) ON DELETE CASCADE,
                    FOREIGN KEY (source_id) REFERENCES promotion_sources(id) ON DELETE SET NULL
                );
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotions_state ON promotions(verification_state);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotions_target ON promotions(target_id);"
            )

    def ensure_target(self, domain: str) -> int:
        """Create or refresh a target anchor row and return its ID."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        normalized = domain.strip().lower()
        conn = self.get_connection()
        now = self._utc_now_iso()
        with conn:
            conn.execute(
                """
                INSERT INTO bonus_targets (domain, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET updated_at = EXCLUDED.updated_at;
                """,
                (normalized, now, now),
            )
            row = conn.execute(
                "SELECT id FROM bonus_targets WHERE domain = ?", (normalized,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to resolve target ID for domain: {normalized}")
            return int(row["id"])

    def record_source(
        self,
        domain: str,
        source_url: str,
        raw_content: Union[str, bytes],
        captured_at: Optional[datetime.datetime] = None,
        extraction_method: str = "automated",
    ) -> Tuple[int, bool]:
        """Persist a raw promotional source snapshot with content versioning.

        Returns:
            ``(source_id, changed)`` where ``changed`` is True only when the
            fetched content differs from the latest stored version.
        """
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        raw_text = self._decode_raw_content(raw_content)
        content_hash = SourceProvenance.compute_content_hash(raw_text)
        target_id = self.ensure_target(domain)
        captured_iso = (
            captured_at.astimezone(datetime.timezone.utc).isoformat()
            if isinstance(captured_at, datetime.datetime)
            else self._utc_now_iso()
        )
        method = (
            extraction_method
            if isinstance(extraction_method, str) and extraction_method.strip()
            else "automated"
        )

        conn = self.get_connection()
        previous_version: Optional[int] = None
        with conn:
            row = conn.execute(
                """
                SELECT id, version, content_hash FROM promotion_sources
                WHERE target_id = ? AND source_url = ?
                ORDER BY version DESC LIMIT 1
                """,
                (target_id, source_url),
            ).fetchone()
            if row is not None and row["content_hash"] == content_hash:
                return int(row["id"]), False
            previous_version = int(row["version"]) if row is not None else None
            next_version = previous_version + 1 if previous_version is not None else 1
            cursor = conn.execute(
                """
                INSERT INTO promotion_sources (
                    target_id, source_url, content_hash, raw_content,
                    version, captured_at, extraction_method
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (target_id, source_url, content_hash, raw_text, next_version, captured_iso, method),
            )
            source_id = int(cursor.lastrowid) if cursor.lastrowid is not None else 0
        if previous_version is not None:
            logging.info(
                f"[BONUS_DB] New content version {next_version} recorded for {source_url}"
            )
        return source_id, True

    def has_content_changed(
        self, domain: str, source_url: str, raw_content: Union[str, bytes]
    ) -> bool:
        """Return True when content differs from the latest stored version."""
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        raw_text = self._decode_raw_content(raw_content)
        content_hash = SourceProvenance.compute_content_hash(raw_text)
        conn = self.get_connection()
        row = conn.execute(
            """
            SELECT s.content_hash FROM promotion_sources s
            JOIN bonus_targets t ON s.target_id = t.id
            WHERE t.domain = ? AND s.source_url = ?
            ORDER BY s.version DESC LIMIT 1
            """,
            (domain.strip().lower(), source_url),
        ).fetchone()
        return row is None or row["content_hash"] != content_hash

    def get_latest_source(self, domain: str, source_url: str) -> Optional[Dict[str, Any]]:
        """Return the latest stored source snapshot for a domain and URL."""
        conn = self.get_connection()
        row = conn.execute(
            """
            SELECT s.* FROM promotion_sources s
            JOIN bonus_targets t ON s.target_id = t.id
            WHERE t.domain = ? AND s.source_url = ?
            ORDER BY s.version DESC LIMIT 1
            """,
            (domain.strip().lower(), source_url),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_source_versions(self, domain: str, source_url: str) -> List[Dict[str, Any]]:
        """Return all stored versions of a source, newest first."""
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT s.* FROM promotion_sources s
            JOIN bonus_targets t ON s.target_id = t.id
            WHERE t.domain = ? AND s.source_url = ?
            ORDER BY s.version DESC
            """,
            (domain.strip().lower(), source_url),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_promotion(
        self,
        domain: str,
        promotion: Promotion,
        source_id: Optional[int] = None,
    ) -> str:
        """Upsert a promotion for a target domain; returns the promotion_id."""
        if not isinstance(promotion, Promotion):
            raise TypeError("promotion must be a Promotion instance")
        violations = promotion.validate_invariants()
        if violations:
            logging.warning(
                f"[BONUS_DB] Promotion {promotion.promotion_id} has invariant "
                f"violations: {violations}"
            )
        target_id = self.ensure_target(domain)
        now = self._utc_now_iso()
        provenance = promotion.provenance
        source_url = provenance.source_url if provenance is not None else None
        extraction_method = (
            provenance.extraction_method if provenance is not None else "automated"
        )
        promotion_json = json.dumps(promotion.to_dict())

        conn = self.get_connection()
        query = """
            INSERT INTO promotions (
                target_id, promotion_id, promotion_type, title, verification_state,
                source_id, source_url, extraction_method, promotion_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(promotion_id) DO UPDATE SET
                promotion_type = EXCLUDED.promotion_type,
                title = EXCLUDED.title,
                verification_state = EXCLUDED.verification_state,
                source_id = EXCLUDED.source_id,
                source_url = EXCLUDED.source_url,
                extraction_method = EXCLUDED.extraction_method,
                promotion_json = EXCLUDED.promotion_json,
                updated_at = EXCLUDED.updated_at;
        """
        with conn:
            conn.execute(
                query,
                (
                    target_id,
                    promotion.promotion_id,
                    promotion.promotion_type.value,
                    promotion.title,
                    promotion.verification_state.value,
                    source_id,
                    source_url,
                    extraction_method,
                    promotion_json,
                    now,
                    now,
                ),
            )
        logging.info(
            f"[BONUS_DB] Persisted promotion {promotion.promotion_id} for domain {domain}"
        )
        return promotion.promotion_id

    @staticmethod
    def _row_to_promotion(row: sqlite3.Row) -> Promotion:
        try:
            payload = json.loads(row["promotion_json"])
        except (json.JSONDecodeError, TypeError) as err:
            raise ValueError(
                f"Corrupted promotion payload for promotion_id: {row['promotion_id']}"
            ) from err
        if not isinstance(payload, dict):
            raise ValueError(
                f"Corrupted promotion payload for promotion_id: {row['promotion_id']}"
            )
        return Promotion.from_dict(payload)

    def get_promotion(self, promotion_id: str) -> Optional[Promotion]:
        """Fetch a single promotion by its canonical promotion_id."""
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise ValueError("promotion_id must be a non-empty string")
        conn = self.get_connection()
        row = conn.execute(
            "SELECT * FROM promotions WHERE promotion_id = ?", (promotion_id,)
        ).fetchone()
        return self._row_to_promotion(row) if row is not None else None

    def get_promotions_by_domain(
        self, domain: str, state: Optional[VerificationState] = None
    ) -> List[Promotion]:
        """Fetch all promotions for a domain, optionally filtered by state."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        conn = self.get_connection()
        if state is not None:
            state_value = (
                state.value
                if isinstance(state, VerificationState)
                else VerificationState(state).value
            )
            rows = conn.execute(
                """
                SELECT p.* FROM promotions p
                JOIN bonus_targets t ON p.target_id = t.id
                WHERE t.domain = ? AND p.verification_state = ?
                """,
                (domain.strip().lower(), state_value),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT p.* FROM promotions p
                JOIN bonus_targets t ON p.target_id = t.id
                WHERE t.domain = ?
                """,
                (domain.strip().lower(),),
            ).fetchall()
        return [self._row_to_promotion(r) for r in rows]

    def get_promotions_by_state(self, state: VerificationState) -> List[Promotion]:
        """Fetch all promotions currently in a given lifecycle state."""
        state_value = (
            state.value
            if isinstance(state, VerificationState)
            else VerificationState(state).value
        )
        conn = self.get_connection()
        rows = conn.execute(
            "SELECT * FROM promotions WHERE verification_state = ?", (state_value,)
        ).fetchall()
        return [self._row_to_promotion(r) for r in rows]

    def get_promotions_by_provenance(
        self,
        source_url: Optional[str] = None,
        extraction_method: Optional[str] = None,
    ) -> List[Promotion]:
        """Fetch promotions matching provenance filters (at least one required)."""
        if source_url is None and extraction_method is None:
            raise ValueError("At least one provenance filter must be provided")
        conn = self.get_connection()
        conditions: List[str] = []
        params: List[Any] = []
        if source_url is not None:
            conditions.append("source_url = ?")
            params.append(source_url)
        if extraction_method is not None:
            conditions.append("extraction_method = ?")
            params.append(extraction_method)
        where_clause = " WHERE " + " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM promotions{where_clause}", tuple(params)
        ).fetchall()
        return [self._row_to_promotion(r) for r in rows]

    def transition_promotion(
        self, promotion_id: str, next_state: VerificationState
    ) -> Promotion:
        """Atomically advance a promotion's lifecycle state.

        State-machine validation and persistence happen inside a single
        ``BEGIN IMMEDIATE`` transaction; on any failure the store is left
        untouched and the original exception propagates.
        """
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise ValueError("promotion_id must be a non-empty string")
        conn = self.get_connection()
        now = self._utc_now_iso()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT promotion_json FROM promotions WHERE promotion_id = ?",
                (promotion_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown promotion_id: {promotion_id!r}")
            promotion = self._row_to_promotion(row)
            promotion.transition_to(next_state)
            conn.execute(
                """
                UPDATE promotions
                SET verification_state = ?, promotion_json = ?, updated_at = ?
                WHERE promotion_id = ?
                """,
                (
                    promotion.verification_state.value,
                    json.dumps(promotion.to_dict()),
                    now,
                    promotion_id,
                ),
            )
            conn.commit()
            logging.info(
                f"[BONUS_DB] Promotion {promotion_id} transitioned to "
                f"{promotion.verification_state.value}"
            )
            return promotion
        except Exception:
            conn.rollback()
            raise

    def apply_operator_adjustment(
        self,
        promotion_id: str,
        updates: Dict[str, Any],
        operator_note: Optional[str] = None,
    ) -> Promotion:
        """Atomically apply operator review adjustments and confirm a promotion.

        Adjustment keys are validated dynamically against the serialized
        ``Promotion`` fields (no hardcoded field vocabulary). The promotion
        is transitioned to ``OPERATOR_CONFIRMED`` — routing through
        ``INVARIANT_VALIDATED`` when still ``EXTRACTED`` — within the same
        atomic transaction.
        """
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise ValueError("promotion_id must be a non-empty string")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates must be a non-empty dictionary")
        conn = self.get_connection()
        now = self._utc_now_iso()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT promotion_json FROM promotions WHERE promotion_id = ?",
                (promotion_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown promotion_id: {promotion_id!r}")
            promotion = self._row_to_promotion(row)
            current = promotion.to_dict()
            unknown_keys = set(updates) - set(current)
            if unknown_keys:
                raise ValueError(f"Unknown adjustment fields: {sorted(unknown_keys)}")
            merged = dict(current)
            merged.update(updates)
            if isinstance(operator_note, str) and operator_note.strip():
                merged["notes"] = operator_note.strip()
            adjusted = Promotion.from_dict(merged)
            if adjusted.verification_state == VerificationState.EXTRACTED:
                adjusted.transition_to(VerificationState.INVARIANT_VALIDATED)
            adjusted.transition_to(VerificationState.OPERATOR_CONFIRMED)
            conn.execute(
                """
                UPDATE promotions
                SET verification_state = ?, promotion_json = ?, updated_at = ?
                WHERE promotion_id = ?
                """,
                (
                    adjusted.verification_state.value,
                    json.dumps(adjusted.to_dict()),
                    now,
                    promotion_id,
                ),
            )
            conn.commit()
            logging.info(
                f"[BONUS_DB] Operator adjustments applied to promotion {promotion_id}"
            )
            return adjusted
        except Exception:
            conn.rollback()
            raise
