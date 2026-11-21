"""SQLite persistence layer for bonus_engine promotional domain models.

This store is fully isolated from the control-plane database: it owns its
own SQLite file, schema, and connection lifecycle. It persists the
relationship between target domains, raw promotional sources (with content
versioning and change detection), and extracted campaigns.

Concurrency model: a single re-entrant lock guards the shared connection
and every transaction, so concurrent background worker threads can never
interleave multi-statement writes or corrupt SQLite state.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import pathlib
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from bonus_engine.models import (
    Promotion,
    SourceProvenance,
    TargetAcquisitionState,
    VerificationState,
    find_transition_path,
)

logger = logging.getLogger(__name__)


class BonusDatabaseManager:
    """Dedicated persistence layer for promotions, sources, and targets."""

    _OPERATOR_PROTECTED_STATES = frozenset(
        {
            VerificationState.OPERATOR_CONFIRMED,
            VerificationState.PENDING_APPROVAL,
            VerificationState.APPROVED,
        }
    )

    def __init__(self, db_path: Union[str, pathlib.Path] = "./data/bonus_engine.db") -> None:
        self.db_path: pathlib.Path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(
                    str(self.db_path), check_same_thread=False, isolation_level=None
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA busy_timeout=5000;")
                self._conn = conn
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as err:
                    logger.warning(f"[BONUS_DB] Error closing database connection: {err}")
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

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize and isolate one atomic multi-statement write transaction."""
        with self._lock:
            conn = self.get_connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _init_db(self) -> None:
        with self._transaction() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS bonus_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    acquisition_state TEXT NOT NULL DEFAULT '{TargetAcquisitionState.DISCOVERED.value}'
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS target_source_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_id INTEGER NOT NULL,
                    source_url TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_content_hash TEXT,
                    UNIQUE(target_id, source_url),
                    FOREIGN KEY (target_id) REFERENCES bonus_targets(id) ON DELETE CASCADE
                );
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotions_state ON promotions(verification_state);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_promotions_target ON promotions(target_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_registry_target ON target_source_registry(target_id);"
            )
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Idempotently align acquisition tracking with canonical enum values.

        Adds the ``acquisition_state`` column to pre-existing databases using
        the canonical default, and rewrites any persisted value outside the
        ``TargetAcquisitionState`` vocabulary so state deserialization never
        fails on legacy rows.
        """
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(bonus_targets)")
        }
        default_state = TargetAcquisitionState.DISCOVERED.value
        if "acquisition_state" not in existing_columns:
            # SQLite does not accept bound parameters in ALTER TABLE; the
            # interpolated value is the validated enum identifier, never
            # user input.
            conn.execute(
                "ALTER TABLE bonus_targets "
                f"ADD COLUMN acquisition_state TEXT NOT NULL DEFAULT '{default_state}';"
            )
        valid_states = tuple(state.value for state in TargetAcquisitionState)
        placeholders = ", ".join("?" for _ in valid_states)
        cursor = conn.execute(
            "UPDATE bonus_targets SET acquisition_state = ? "
            f"WHERE acquisition_state IS NULL "
            f"OR acquisition_state NOT IN ({placeholders})",
            (default_state, *valid_states),
        )
        if cursor.rowcount:
            logger.warning(
                "[BONUS_DB] Normalized %d bonus_targets row(s) with invalid "
                "acquisition_state to '%s'",
                cursor.rowcount,
                default_state,
            )

    @staticmethod
    def _coerce_acquisition_state(value: Union[TargetAcquisitionState, str]) -> str:
        if isinstance(value, TargetAcquisitionState):
            return value.value
        if isinstance(value, str):
            try:
                return TargetAcquisitionState(value.strip().lower()).value
            except ValueError:
                raise ValueError(f"Unknown target acquisition state: {value!r}") from None
        raise TypeError("acquisition_state must be a TargetAcquisitionState or string")

    def ensure_target(
        self,
        domain: str,
        acquisition_state: Optional[Union[TargetAcquisitionState, str]] = None,
    ) -> int:
        """Create or refresh a target anchor row and return its ID.

        New targets default to ``DISCOVERED``; existing targets keep their
        persisted acquisition state unless one is explicitly supplied.
        """
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        normalized = domain.strip().lower()
        state_value = (
            self._coerce_acquisition_state(acquisition_state)
            if acquisition_state is not None
            else TargetAcquisitionState.DISCOVERED.value
        )
        now = self._utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO bonus_targets (domain, created_at, updated_at, acquisition_state)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    acquisition_state = COALESCE(?, bonus_targets.acquisition_state);
                """,
                (
                    normalized,
                    now,
                    now,
                    state_value,
                    state_value if acquisition_state is not None else None,
                ),
            )
            row = conn.execute(
                "SELECT id FROM bonus_targets WHERE domain = ?", (normalized,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Failed to resolve target ID for domain: {normalized}")
            return int(row["id"])

    def set_target_acquisition_state(
        self,
        domain: str,
        state: Union[TargetAcquisitionState, str],
    ) -> None:
        """Create-or-update a target and persist its acquisition state atomically."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        state_value = self._coerce_acquisition_state(state)
        normalized = domain.strip().lower()
        now = self._utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO bonus_targets (domain, created_at, updated_at, acquisition_state)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    acquisition_state = EXCLUDED.acquisition_state;
                """,
                (normalized, now, now, state_value),
            )

    def get_target_acquisition_state(
        self, domain: str
    ) -> Optional[TargetAcquisitionState]:
        """Return the persisted acquisition state for a domain, if known."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        with self._lock:
            row = self.get_connection().execute(
                "SELECT acquisition_state FROM bonus_targets WHERE domain = ?",
                (domain.strip().lower(),),
            ).fetchone()
        if row is None or row["acquisition_state"] is None:
            return None
        try:
            return TargetAcquisitionState(row["acquisition_state"])
        except ValueError as err:
            raise ValueError(
                f"Corrupted acquisition_state for domain: {domain.strip().lower()}"
            ) from err

    def get_targets_by_acquisition_state(
        self, state: Union[TargetAcquisitionState, str]
    ) -> List[str]:
        """Return all domains currently in the given acquisition state."""
        state_value = self._coerce_acquisition_state(state)
        with self._lock:
            rows = self.get_connection().execute(
                "SELECT domain FROM bonus_targets WHERE acquisition_state = ?",
                (state_value,),
            ).fetchall()
        return [str(row["domain"]) for row in rows]

    def register_source_url(
        self,
        domain: str,
        source_url: str,
        pinned: bool = False,
    ) -> None:
        """Register (or re-pin) a source URL for periodic diff monitoring."""
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        target_id = self.ensure_target(domain)
        now = self._utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO target_source_registry (
                    target_id, source_url, pinned, registered_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(target_id, source_url) DO UPDATE SET
                    pinned = EXCLUDED.pinned,
                    updated_at = EXCLUDED.updated_at;
                """,
                (target_id, source_url.strip(), 1 if pinned else 0, now, now),
            )

    def unregister_source_url(self, domain: str, source_url: str) -> bool:
        """Remove a registered source URL; returns True when a row was removed."""
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                DELETE FROM target_source_registry
                WHERE target_id = (SELECT id FROM bonus_targets WHERE domain = ?)
                  AND source_url = ?
                """,
                (domain.strip().lower(), source_url.strip()),
            )
        return cursor.rowcount > 0

    def get_registered_source_urls(
        self, domain: str, pinned_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List registered source URLs for a domain with diff-watcher metadata."""
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("domain must be a non-empty string")
        query = """
            SELECT r.source_url, r.pinned, r.registered_at, r.updated_at,
                   r.last_checked_at, r.last_content_hash
            FROM target_source_registry r
            JOIN bonus_targets t ON r.target_id = t.id
            WHERE t.domain = ?
        """
        params: List[Any] = [domain.strip().lower()]
        if pinned_only:
            query += " AND r.pinned = 1"
        query += " ORDER BY r.registered_at ASC"
        with self._lock:
            rows = self.get_connection().execute(query, tuple(params)).fetchall()
        return [
            {
                "source_url": row["source_url"],
                "pinned": bool(row["pinned"]),
                "registered_at": row["registered_at"],
                "updated_at": row["updated_at"],
                "last_checked_at": row["last_checked_at"],
                "last_content_hash": row["last_content_hash"],
            }
            for row in rows
        ]

    def record_source_check(
        self,
        domain: str,
        source_url: str,
        content_hash: Optional[str] = None,
    ) -> bool:
        """Record a diff-watcher probe result for a registered source URL.

        Returns True when the URL was registered; False when unknown.
        """
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("source_url must be a non-empty string")
        now = self._utc_now_iso()
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE target_source_registry
                SET last_checked_at = ?,
                    last_content_hash = COALESCE(?, last_content_hash),
                    updated_at = ?
                WHERE target_id = (SELECT id FROM bonus_targets WHERE domain = ?)
                  AND source_url = ?
                """,
                (now, content_hash, now, domain.strip().lower(), source_url.strip()),
            )
        return cursor.rowcount > 0

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

        previous_version: Optional[int] = None
        with self._transaction() as conn:
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
            logger.info(
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        """Upsert a promotion, protecting operator-verified records.

        Promotions in operator-protected states are never overwritten unless a
        new ``source_id`` proves the upstream content changed; in that case the
        verified record is preserved and the re-extraction is persisted as a new
        revision starting from ``EXTRACTED``.
        """
        if not isinstance(promotion, Promotion):
            raise TypeError("promotion must be a Promotion instance")
        violations = promotion.validate_invariants()
        if violations:
            logger.warning(
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
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT source_id, verification_state FROM promotions "
                "WHERE promotion_id = ?",
                (promotion.promotion_id,),
            ).fetchone()
            if existing is not None:
                existing_state = VerificationState(existing["verification_state"])
                if existing_state in self._OPERATOR_PROTECTED_STATES:
                    existing_source_id = existing["source_id"]
                    content_changed = (
                        source_id is not None
                        and existing_source_id is not None
                        and int(existing_source_id) != int(source_id)
                    )
                    if not content_changed:
                        logger.info(
                            "[BONUS_DB] Skipping update of operator-protected "
                            f"promotion {promotion.promotion_id} "
                            f"(state={existing_state.value}, no upstream content change)"
                        )
                        return promotion.promotion_id
                    revision_payload = promotion.to_dict()
                    revision_payload["promotion_id"] = hashlib.sha256(
                        f"{promotion.promotion_id}{uuid.uuid4().hex}"
                        f"{self._utc_now_iso()}".encode()
                    ).hexdigest()
                    revision_payload["verification_state"] = (
                        VerificationState.EXTRACTED.value
                    )
                    revision = Promotion.from_dict(revision_payload)
                    conn.execute(
                        """
                        INSERT INTO promotions (
                            target_id, promotion_id, promotion_type, title,
                            verification_state, source_id, source_url,
                            extraction_method, promotion_json, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            target_id,
                            revision.promotion_id,
                            revision.promotion_type.value,
                            revision.title,
                            revision.verification_state.value,
                            source_id,
                            source_url,
                            extraction_method,
                            json.dumps(revision.to_dict()),
                            now,
                            now,
                        ),
                    )
                    logger.info(
                        "[BONUS_DB] Upstream content changed for operator-protected "
                        f"promotion {promotion.promotion_id}; preserved verified "
                        f"record and stored revision {revision.promotion_id}"
                    )
                    return revision.promotion_id
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
        logger.info(
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        now = self._utc_now_iso()
        with self._transaction() as conn:
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
        logger.info(
            f"[BONUS_DB] Promotion {promotion_id} transitioned to "
            f"{promotion.verification_state.value}"
        )
        return promotion

    def apply_operator_adjustment(
        self,
        promotion_id: str,
        updates: Dict[str, Any],
        operator_note: Optional[str] = None,
    ) -> Promotion:
        """Atomically apply operator review adjustments and confirm a promotion.

        Adjustment keys are validated dynamically against the serialized
        ``Promotion`` fields (no hardcoded field vocabulary). The promotion
        is routed to ``OPERATOR_CONFIRMED`` along the shortest legal transition
        path computed from the canonical state machine (empty path when already
        confirmed) within the same atomic transaction.
        """
        if not isinstance(promotion_id, str) or not promotion_id.strip():
            raise ValueError("promotion_id must be a non-empty string")
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates must be a non-empty dictionary")
        now = self._utc_now_iso()
        with self._transaction() as conn:
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
            for step in find_transition_path(
                adjusted.verification_state, VerificationState.OPERATOR_CONFIRMED
            ):
                adjusted.transition_to(step)
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
        logger.info(
            f"[BONUS_DB] Operator adjustments applied to promotion {promotion_id}"
        )
        return adjusted
