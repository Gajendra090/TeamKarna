"""SQLite persistence layer for KNOWLEDGE X Research History."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "knowledge_x_history.db"
LEGACY_USER_ID = "legacy_local"
_WAL_LOCK = threading.Lock()
_WAL_CONFIGURED_PATHS: set[str] = set()

_SECRET_KEY_PATTERN = re.compile(r"(api[_\-]?key|secret|token|password|credential)", re.IGNORECASE)
_GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{16,}")


class HistoryStorageError(Exception):
    """Raised when SQLite history storage operations fail."""


def _normalize_user_id(user_id: str | None) -> str:
    cleaned = (user_id or "").strip()
    return cleaned if cleaned else LEGACY_USER_ID


def _sanitize_for_storage(value: Any) -> Any:
    """Recursively scrub API keys or secret-like fields before writing to SQLite."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if _SECRET_KEY_PATTERN.search(key_str):
                continue
            cleaned[key_str] = _sanitize_for_storage(v)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_for_storage(item) for item in value]
    if isinstance(value, str):
        return _GOOGLE_KEY_PATTERN.sub("[REDACTED_KEY]", value)
    return value


@contextmanager
def _connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _ensure_wal_once(conn: sqlite3.Connection, db_path: Path | str) -> None:
    """Configure SQLite WAL journal mode once per database path without repeating on every connect."""
    resolved = str(Path(db_path).resolve())
    with _WAL_LOCK:
        if resolved in _WAL_CONFIGURED_PATHS:
            return
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _WAL_CONFIGURED_PATHS.add(resolved)


def init_history_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Ensure SQLite history tables, user_id columns, and indexes exist safely."""
    try:
        with _connect(db_path) as conn:
            _ensure_wal_once(conn, db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'legacy_local',
                    created_at TEXT NOT NULL,
                    research_topic TEXT NOT NULL,
                    document_names_json TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    analysis_results_json TEXT NOT NULL,
                    common_concepts_json TEXT NOT NULL,
                    research_gaps_json TEXT NOT NULL,
                    knowledge_graph_json TEXT NOT NULL
                )
                """
            )
            _ensure_column(
                conn,
                "research_sessions",
                "user_id",
                "TEXT NOT NULL DEFAULT 'legacy_local'",
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_research_sessions_created_at
                ON research_sessions (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_research_sessions_user
                ON research_sessions (user_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ambiguity_history (
                    record_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'legacy_local',
                    created_at TEXT NOT NULL,
                    document_names_json TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    unclear_phrases_json TEXT NOT NULL,
                    ambiguity_reason TEXT NOT NULL,
                    interpretations_json TEXT NOT NULL,
                    user_clarification TEXT NOT NULL,
                    resolved_question TEXT NOT NULL,
                    final_answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    model_used TEXT NOT NULL
                )
                """
            )
            _ensure_column(
                conn,
                "ambiguity_history",
                "user_id",
                "TEXT NOT NULL DEFAULT 'legacy_local'",
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ambiguity_history_created_at
                ON ambiguity_history (created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ambiguity_history_user
                ON ambiguity_history (user_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    file_size_kb REAL NOT NULL,
                    page_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section_title TEXT NOT NULL DEFAULT '',
                    chunk_text TEXT NOT NULL,
                    chunk_word_count INTEGER NOT NULL,
                    chunk_char_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_file_hash
                ON document_chunks (file_hash, chunk_index ASC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_page
                ON document_chunks (file_hash, page_number ASC, chunk_index ASC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_filename
                ON document_chunks (filename)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS multi_llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    worker_name TEXT NOT NULL,
                    processing_version TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    token_usage_json TEXT NOT NULL,
                    error_message TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_multi_llm_cache_lookup
                ON multi_llm_cache (file_hash, worker_name, processing_version)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    embedding_key TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    embedding_vector TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_file_model
                ON chunk_embeddings (file_hash, embedding_model)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk_id
                ON chunk_embeddings (chunk_id)
                """
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to initialize history database: {exc}") from exc


def derive_research_topic(
    documents: list[dict[str, Any]],
    analysis_result: dict[str, Any] | None = None,
) -> str:
    """Generate a concise, informative research topic label for a completed session."""
    if analysis_result:
        concepts = [
            c.get("concept", "").strip()
            for c in analysis_result.get("common_concepts", [])
            if c.get("concept", "").strip()
        ]
        if concepts:
            topic = " · ".join(concepts[:3])
            return topic[:110]

        overview = (analysis_result.get("corpus_overview") or "").strip()
        if overview:
            first_sentence = re.split(r"(?<=[.!?])\s+", overview)[0].strip()
            return first_sentence[:110]

    titles = [
        (d.get("title") or d.get("filename") or "Document").strip()
        for d in (documents or [])
    ]
    if titles:
        joined = ", ".join(titles[:2])
        if len(titles) > 2:
            joined += f" (+{len(titles) - 2} more)"
        return joined[:110]
    return "Untitled Research Session"


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    """Deserialize a SQLite row into a structured Python dictionary."""
    def _loads(raw: str, default: Any) -> Any:
        try:
            return json.loads(raw) if raw else default
        except json.JSONDecodeError:
            return default

    return {
        "session_id": row["session_id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else LEGACY_USER_ID,
        "created_at": row["created_at"],
        "research_topic": row["research_topic"],
        "document_names": _loads(row["document_names_json"], []),
        "document_count": int(row["document_count"]),
        "analysis_results": _loads(row["analysis_results_json"], {}),
        "common_concepts": _loads(row["common_concepts_json"], []),
        "research_gaps": _loads(row["research_gaps_json"], {}),
        "knowledge_graph": _loads(row["knowledge_graph_json"], {}),
    }


def save_research_session(
    documents: list[dict[str, Any]],
    analysis_results: dict[str, Any] | None = None,
    research_gaps: dict[str, Any] | None = None,
    knowledge_graph: dict[str, Any] | None = None,
    research_topic: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Persist or update a user-scoped research session in SQLite and return its session_id."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sid = session_id or f"KX-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    doc_names = [str(d.get("filename", "document.pdf")) for d in (documents or [])]
    if not doc_names and analysis_results:
        doc_names = [
            str(item.get("filename", ""))
            for item in analysis_results.get("document_analyses", [])
            if item.get("filename")
        ]

    safe_analysis = _sanitize_for_storage(analysis_results or {})
    safe_concepts = _sanitize_for_storage(
        safe_analysis.get("common_concepts", []) if isinstance(safe_analysis, dict) else []
    )
    safe_gaps = _sanitize_for_storage(research_gaps or {})
    safe_graph = _sanitize_for_storage(knowledge_graph or {})

    topic = (research_topic or "").strip() or derive_research_topic(documents, safe_analysis)
    topic = _sanitize_for_storage(topic)

    try:
        with _connect(db_path) as conn:
            existing_any = conn.execute(
                "SELECT session_id, user_id, created_at, analysis_results_json, common_concepts_json, "
                "research_gaps_json, knowledge_graph_json FROM research_sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()

            if existing_any and existing_any["user_id"] != owner_id:
                # Never overwrite another user's session_id
                sid = f"KX-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
                existing_any = None

            if existing_any:
                if not safe_analysis:
                    safe_analysis = json.loads(existing_any["analysis_results_json"] or "{}")
                    safe_concepts = json.loads(existing_any["common_concepts_json"] or "[]")
                if not safe_gaps:
                    safe_gaps = json.loads(existing_any["research_gaps_json"] or "{}")
                if not safe_graph:
                    safe_graph = json.loads(existing_any["knowledge_graph_json"] or "{}")

                conn.execute(
                    """
                    UPDATE research_sessions
                    SET research_topic = ?,
                        document_names_json = ?,
                        document_count = ?,
                        analysis_results_json = ?,
                        common_concepts_json = ?,
                        research_gaps_json = ?,
                        knowledge_graph_json = ?
                    WHERE session_id = ? AND user_id = ?
                    """,
                    (
                        topic,
                        json.dumps(doc_names),
                        len(doc_names),
                        json.dumps(safe_analysis),
                        json.dumps(safe_concepts),
                        json.dumps(safe_gaps),
                        json.dumps(safe_graph),
                        sid,
                        owner_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO research_sessions (
                        session_id,
                        user_id,
                        created_at,
                        research_topic,
                        document_names_json,
                        document_count,
                        analysis_results_json,
                        common_concepts_json,
                        research_gaps_json,
                        knowledge_graph_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        owner_id,
                        now_iso,
                        topic,
                        json.dumps(doc_names),
                        len(doc_names),
                        json.dumps(safe_analysis),
                        json.dumps(safe_concepts),
                        json.dumps(safe_gaps),
                        json.dumps(safe_graph),
                    ),
                )
            conn.commit()
        return sid
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to save research session: {exc}") from exc


def list_research_sessions(
    search_query: str = "",
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return user-scoped research sessions ordered chronologically (newest first)."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)

    clauses: list[str] = ["user_id = ?"]
    params: list[Any] = [owner_id]

    if date_from:
        start_str = date_from.strftime("%Y-%m-%d") if isinstance(date_from, date) else str(date_from)[:10]
        clauses.append("created_at >= ?")
        params.append(f"{start_str} 00:00:00")

    if date_to:
        end_str = date_to.strftime("%Y-%m-%d") if isinstance(date_to, date) else str(date_to)[:10]
        clauses.append("created_at <= ?")
        params.append(f"{end_str} 23:59:59")

    q = (search_query or "").strip()
    if q:
        like_q = f"%{q.lower()}%"
        clauses.append(
            "("
            "LOWER(research_topic) LIKE ? OR "
            "LOWER(document_names_json) LIKE ? OR "
            "LOWER(common_concepts_json) LIKE ? OR "
            "LOWER(analysis_results_json) LIKE ?"
            ")"
        )
        params.extend([like_q, like_q, like_q, like_q])

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query_sql = f"SELECT * FROM research_sessions {where_sql} ORDER BY created_at DESC"

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(query_sql, params).fetchall()
        return [_row_to_session(row) for row in rows]
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to query research history: {exc}") from exc


def get_research_session(
    session_id: str,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Fetch a single saved research session scoped to user_id."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM research_sessions WHERE session_id = ? AND user_id = ?",
                (session_id, owner_id),
            ).fetchone()
        return _row_to_session(row) if row else None
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load session '{session_id}': {exc}") from exc


def delete_research_session(
    session_id: str,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Delete a single research session record scoped to user_id."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)
    try:
        with _connect(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM research_sessions WHERE session_id = ? AND user_id = ?",
                (session_id, owner_id),
            )
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to delete session '{session_id}': {exc}") from exc


def _row_to_ambiguity(row: sqlite3.Row) -> dict[str, Any]:
    def _loads(raw: str, default: Any) -> Any:
        try:
            return json.loads(raw) if raw else default
        except json.JSONDecodeError:
            return default

    return {
        "record_id": row["record_id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else LEGACY_USER_ID,
        "created_at": row["created_at"],
        "document_names": _loads(row["document_names_json"], []),
        "original_question": row["original_question"],
        "classification": row["classification"],
        "unclear_phrases": _loads(row["unclear_phrases_json"], []),
        "ambiguity_reason": row["ambiguity_reason"],
        "interpretations": _loads(row["interpretations_json"], []),
        "user_clarification": row["user_clarification"],
        "resolved_question": row["resolved_question"],
        "final_answer": row["final_answer"],
        "citations": _loads(row["citations_json"], []),
        "model_used": row["model_used"],
    }


def save_ambiguity_record(
    documents: list[dict[str, Any]],
    original_question: str,
    classification_info: dict[str, Any],
    user_clarification: str,
    resolved_question: str,
    final_answer: str,
    citations: list[dict[str, Any]] | None = None,
    model_used: str = "",
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Persist a user-scoped ambiguity detection and resolution record in SQLite."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = f"AMB-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    doc_names = [str(d.get("filename", "document.pdf")) for d in (documents or [])]
    safe_orig = _sanitize_for_storage(str(original_question or "").strip())
    safe_cls = str(classification_info.get("classification", "Clear"))
    safe_phrases = _sanitize_for_storage(classification_info.get("unclear_phrases", []))
    safe_reason = _sanitize_for_storage(str(classification_info.get("ambiguity_reason", "")))
    safe_interps = _sanitize_for_storage(classification_info.get("interpretations", []))
    safe_clar = _sanitize_for_storage(str(user_clarification or "").strip())
    safe_resolved = _sanitize_for_storage(str(resolved_question or safe_orig).strip())
    safe_answer = _sanitize_for_storage(str(final_answer or "").strip())
    safe_citations = _sanitize_for_storage(citations or [])
    safe_model = _sanitize_for_storage(str(model_used or "").strip())

    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO ambiguity_history (
                    record_id,
                    user_id,
                    created_at,
                    document_names_json,
                    original_question,
                    classification,
                    unclear_phrases_json,
                    ambiguity_reason,
                    interpretations_json,
                    user_clarification,
                    resolved_question,
                    final_answer,
                    citations_json,
                    model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    owner_id,
                    now_iso,
                    json.dumps(doc_names),
                    safe_orig,
                    safe_cls,
                    json.dumps(safe_phrases),
                    safe_reason,
                    json.dumps(safe_interps),
                    safe_clar,
                    safe_resolved,
                    safe_answer,
                    json.dumps(safe_citations),
                    safe_model,
                ),
            )
            conn.commit()
        return rid
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to save ambiguity resolution: {exc}") from exc


def list_ambiguity_records(
    search_query: str = "",
    classification_filter: str = "All",
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """List user-scoped ambiguity detection and resolution records chronologically."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)

    clauses: list[str] = ["user_id = ?"]
    params: list[Any] = [owner_id]

    if classification_filter and classification_filter != "All":
        clauses.append("classification = ?")
        params.append(classification_filter)

    if date_from:
        start_str = date_from.strftime("%Y-%m-%d") if isinstance(date_from, date) else str(date_from)[:10]
        clauses.append("created_at >= ?")
        params.append(f"{start_str} 00:00:00")

    if date_to:
        end_str = date_to.strftime("%Y-%m-%d") if isinstance(date_to, date) else str(date_to)[:10]
        clauses.append("created_at <= ?")
        params.append(f"{end_str} 23:59:59")

    q = (search_query or "").strip()
    if q:
        like_q = f"%{q.lower()}%"
        clauses.append(
            "("
            "LOWER(original_question) LIKE ? OR "
            "LOWER(resolved_question) LIKE ? OR "
            "LOWER(document_names_json) LIKE ? OR "
            "LOWER(ambiguity_reason) LIKE ?"
            ")"
        )
        params.extend([like_q, like_q, like_q, like_q])

    where_sql = f"WHERE {' AND '.join(clauses)}"
    query_sql = f"SELECT * FROM ambiguity_history {where_sql} ORDER BY created_at DESC"

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(query_sql, params).fetchall()
        return [_row_to_ambiguity(row) for row in rows]
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to query ambiguity history: {exc}") from exc


def delete_ambiguity_record(
    record_id: str,
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Delete a single ambiguity resolution record scoped to user_id."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)
    try:
        with _connect(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM ambiguity_history WHERE record_id = ? AND user_id = ?",
                (record_id, owner_id),
            )
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to delete ambiguity record '{record_id}': {exc}") from exc


def get_user_history_summary(
    user_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Return total saved analyses and ambiguity resolutions for a specific user."""
    init_history_db(db_path)
    owner_id = _normalize_user_id(user_id)
    try:
        with _connect(db_path) as conn:
            s_cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM research_sessions WHERE user_id = ?",
                (owner_id,),
            ).fetchone()["c"]
            a_cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM ambiguity_history WHERE user_id = ?",
                (owner_id,),
            ).fetchone()["c"]
        return {"research_sessions": int(s_cnt), "ambiguity_records": int(a_cnt)}
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load user summary: {exc}") from exc


def save_document_chunks(
    doc_record: dict[str, Any],
    chunks: list[dict[str, Any]],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Persist extracted document chunks in SQLite keyed by SHA-256 file_hash."""
    file_hash = str(doc_record.get("file_hash") or "").strip()
    if not file_hash or not chunks:
        return 0

    init_history_db(db_path)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = str(doc_record.get("filename") or "document.pdf")
    title = str(doc_record.get("title") or filename)
    author = str(doc_record.get("author") or "Unknown")
    subject = str(doc_record.get("subject") or "")
    file_size_kb = float(doc_record.get("file_size_kb") or 0.0)
    page_count = int(doc_record.get("page_count") or 0)
    word_count = int(doc_record.get("word_count") or 0)
    char_count = int(doc_record.get("char_count") or 0)

    try:
        with _connect(db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM document_chunks WHERE file_hash = ? LIMIT 1",
                (file_hash,),
            ).fetchone()
            if existing:
                return 0

            rows = []
            for idx, chunk in enumerate(chunks):
                chunk_index = int(chunk.get("chunk_index", idx))
                chunk_text = str(chunk.get("text") or "")
                rows.append(
                    (
                        f"{file_hash}:{chunk_index}",
                        file_hash,
                        filename,
                        title,
                        author,
                        subject,
                        file_size_kb,
                        page_count,
                        word_count,
                        char_count,
                        int(chunk.get("page_number") or 1),
                        chunk_index,
                        str(chunk.get("section_title") or ""),
                        chunk_text,
                        int(chunk.get("word_count") or len(chunk_text.split())),
                        int(chunk.get("char_count") or len(chunk_text)),
                        created_at,
                    )
                )

            conn.executemany(
                """
                INSERT OR IGNORE INTO document_chunks (
                    chunk_id, file_hash, filename, title, author, subject,
                    file_size_kb, page_count, word_count, char_count,
                    page_number, chunk_index, section_title, chunk_text,
                    chunk_word_count, chunk_char_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return len(rows)
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to persist document chunks: {exc}") from exc


def has_cached_document(
    file_hash: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Return True if extracted chunks for file_hash already exist in SQLite."""
    clean_hash = (file_hash or "").strip()
    if not clean_hash:
        return False
    init_history_db(db_path)
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM document_chunks WHERE file_hash = ? LIMIT 1",
                (clean_hash,),
            ).fetchone()
        return row is not None
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to check cached document '{clean_hash}': {exc}") from exc


def get_document_chunks_by_hash(
    file_hash: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return all cached chunks for a PDF ordered by chunk_index."""
    clean_hash = (file_hash or "").strip()
    if not clean_hash:
        return []
    init_history_db(db_path)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, file_hash, filename, page_number, chunk_index,
                       section_title, chunk_text, chunk_word_count, chunk_char_count, created_at
                FROM document_chunks
                WHERE file_hash = ?
                ORDER BY chunk_index ASC
                """,
                (clean_hash,),
            ).fetchall()
        return [
            {
                "chunk_id": str(r["chunk_id"]),
                "file_hash": str(r["file_hash"]),
                "filename": str(r["filename"]),
                "page_number": int(r["page_number"]),
                "chunk_index": int(r["chunk_index"]),
                "section_title": str(r["section_title"]),
                "text": str(r["chunk_text"]),
                "word_count": int(r["chunk_word_count"]),
                "char_count": int(r["chunk_char_count"]),
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load chunks for '{clean_hash}': {exc}") from exc


def load_cached_document_record(
    file_hash: str,
    filename_override: str | None = None,
    saved_path: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Reconstruct a full document record from cached SQLite chunks without re-parsing PDF."""
    clean_hash = (file_hash or "").strip()
    if not clean_hash:
        return None
    init_history_db(db_path)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, file_hash, filename, title, author, subject,
                       file_size_kb, page_count, word_count, char_count,
                       page_number, chunk_index, section_title, chunk_text,
                       chunk_word_count, chunk_char_count, created_at
                FROM document_chunks
                WHERE file_hash = ?
                ORDER BY chunk_index ASC
                """,
                (clean_hash,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load cached document '{clean_hash}': {exc}") from exc

    if not rows:
        return None

    first = rows[0]
    effective_filename = filename_override or str(first["filename"])
    pages_map: dict[int, list[str]] = {}
    chunks: list[dict[str, Any]] = []

    for r in rows:
        pno = int(r["page_number"])
        ctext = str(r["chunk_text"])
        pages_map.setdefault(pno, []).append(ctext)
        chunks.append(
            {
                "chunk_id": str(r["chunk_id"]),
                "file_hash": str(r["file_hash"]),
                "filename": effective_filename,
                "page_number": pno,
                "chunk_index": int(r["chunk_index"]),
                "section_title": str(r["section_title"]),
                "text": ctext,
                "word_count": int(r["chunk_word_count"]),
                "char_count": int(r["chunk_char_count"]),
            }
        )

    total_pages = max(int(first["page_count"]), max(pages_map.keys(), default=0))
    pages: list[dict[str, Any]] = []
    formatted_pages: list[str] = []
    for pno in range(1, total_pages + 1):
        p_text = "\n\n".join(pages_map.get(pno, [])).strip()
        pages.append(
            {
                "page_number": pno,
                "text": p_text,
                "word_count": len(p_text.split()),
                "char_count": len(p_text),
            }
        )
        if p_text:
            formatted_pages.append(f"[Page {pno}]\n{p_text}")

    full_text = "\n\n".join(formatted_pages).strip()
    return {
        "filename": effective_filename,
        "title": str(first["title"]),
        "author": str(first["author"]),
        "subject": str(first["subject"]),
        "file_hash": str(first["file_hash"]),
        "file_size_kb": float(first["file_size_kb"]),
        "page_count": int(first["page_count"]),
        "word_count": int(first["word_count"]),
        "char_count": int(first["char_count"]),
        "pages": pages,
        "full_text": full_text,
        "saved_path": saved_path,
        "chunks": chunks,
        "from_cache": True,
    }


def save_worker_cache(
    file_hash: str,
    worker_name: str,
    processing_version: str,
    model_used: str,
    status: str,
    result_data: dict[str, Any] | None,
    token_usage: dict[str, int] | None = None,
    error_message: str = "",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Persist a worker result in SQLite without overwriting valid completed cache with failures."""
    clean_hash = (file_hash or "").strip()
    clean_worker = (worker_name or "").strip()
    clean_ver = (processing_version or "").strip()
    if not clean_hash or not clean_worker or not clean_ver:
        return

    init_history_db(db_path)
    cache_key = f"{clean_hash}:{clean_worker}:{clean_ver}"
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_result = _sanitize_for_storage(result_data or {})
    safe_usage = _sanitize_for_storage(token_usage or {})
    safe_err = str(_sanitize_for_storage(error_message or ""))

    try:
        with _connect(db_path) as conn:
            existing = conn.execute(
                "SELECT status FROM multi_llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if existing and existing["status"] == "completed" and status != "completed":
                return

            conn.execute(
                """
                INSERT INTO multi_llm_cache (
                    cache_key, file_hash, worker_name, processing_version,
                    model_used, status, result_json, token_usage_json,
                    error_message, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model_used = excluded.model_used,
                    status = excluded.status,
                    result_json = excluded.result_json,
                    token_usage_json = excluded.token_usage_json,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    clean_hash,
                    clean_worker,
                    clean_ver,
                    str(model_used or ""),
                    str(status or "failed"),
                    json.dumps(safe_result),
                    json.dumps(safe_usage),
                    safe_err,
                    updated_at,
                ),
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to save worker cache '{cache_key}': {exc}") from exc


def load_worker_cache(
    file_hash: str,
    worker_name: str,
    processing_version: str,
    completed_only: bool = True,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Load a cached worker output from SQLite (by default only completed entries)."""
    clean_hash = (file_hash or "").strip()
    clean_worker = (worker_name or "").strip()
    clean_ver = (processing_version or "").strip()
    if not clean_hash or not clean_worker or not clean_ver:
        return None

    init_history_db(db_path)
    cache_key = f"{clean_hash}:{clean_worker}:{clean_ver}"
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM multi_llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        if completed_only and str(row["status"]) != "completed":
            return None
        def _parse_json(raw_txt: str, fallback: Any) -> Any:
            try:
                return json.loads(raw_txt) if raw_txt else fallback
            except Exception:
                return fallback

        return {
            "cache_key": str(row["cache_key"]),
            "file_hash": str(row["file_hash"]),
            "worker_name": str(row["worker_name"]),
            "processing_version": str(row["processing_version"]),
            "model_used": str(row["model_used"]),
            "status": str(row["status"]),
            "result": _parse_json(str(row["result_json"]), {}),
            "token_usage": _parse_json(str(row["token_usage_json"]), {}),
            "error_message": str(row["error_message"]),
            "updated_at": str(row["updated_at"]),
        }
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load worker cache '{cache_key}': {exc}") from exc


def save_chunk_embeddings(
    records: list[dict[str, Any]],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Persist chunk embedding vectors in SQLite without creating duplicates."""
    if not records:
        return 0

    init_history_db(db_path)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for rec in records:
        chunk_id = str(rec.get("chunk_id") or "").strip()
        file_hash = str(rec.get("file_hash") or "").strip()
        model = str(rec.get("embedding_model") or "").strip()
        vector = rec.get("embedding_vector") or []
        if not chunk_id or not file_hash or not model or not vector:
            continue
        dims = int(rec.get("embedding_dimensions") or len(vector))
        rows.append(
            (
                f"{chunk_id}::{model}",
                chunk_id,
                file_hash,
                model,
                dims,
                json.dumps([float(x) for x in vector]),
                created_at,
            )
        )

    if not rows:
        return 0

    try:
        with _connect(db_path) as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO chunk_embeddings (
                    embedding_key, chunk_id, file_hash, embedding_model,
                    embedding_dimensions, embedding_vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return conn.total_changes - before
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to save chunk embeddings: {exc}") from exc


def get_chunk_embeddings(
    file_hashes: list[str],
    embedding_model: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, dict[str, Any]]:
    """Load stored embeddings for the given file hashes and embedding model keyed by chunk_id."""
    clean_hashes = [h.strip() for h in (file_hashes or []) if h and h.strip()]
    clean_model = (embedding_model or "").strip()
    if not clean_hashes or not clean_model:
        return {}

    init_history_db(db_path)
    placeholders = ",".join("?" for _ in clean_hashes)
    sql = (
        f"SELECT chunk_id, file_hash, embedding_model, embedding_dimensions, embedding_vector "
        f"FROM chunk_embeddings "
        f"WHERE embedding_model = ? AND file_hash IN ({placeholders})"
    )
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(sql, [clean_model, *clean_hashes]).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            try:
                vec = json.loads(str(r["embedding_vector"]))
            except Exception:
                continue
            out[str(r["chunk_id"])] = {
                "chunk_id": str(r["chunk_id"]),
                "file_hash": str(r["file_hash"]),
                "embedding_model": str(r["embedding_model"]),
                "embedding_dimensions": int(r["embedding_dimensions"]),
                "vector": [float(x) for x in vec],
            }
        return out
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to load chunk embeddings: {exc}") from exc


def delete_chunk_embeddings_for_hash(
    file_hash: str,
    embedding_model: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Remove cached embeddings for a file_hash (optionally scoped to one model) for re-indexing."""
    clean_hash = (file_hash or "").strip()
    if not clean_hash:
        return 0
    init_history_db(db_path)
    try:
        with _connect(db_path) as conn:
            if embedding_model:
                cur = conn.execute(
                    "DELETE FROM chunk_embeddings WHERE file_hash = ? AND embedding_model = ?",
                    (clean_hash, embedding_model.strip()),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM chunk_embeddings WHERE file_hash = ?",
                    (clean_hash,),
                )
            conn.commit()
            return int(cur.rowcount or 0)
    except sqlite3.Error as exc:
        raise HistoryStorageError(f"Failed to delete chunk embeddings: {exc}") from exc

