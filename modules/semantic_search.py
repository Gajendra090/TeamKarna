"""Phase 3: Semantic Vector Search and Hybrid Retrieval (RRF) over SQLite chunk_embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modules.ai_analysis import DEFAULT_EMBEDDING_MODEL, generate_embeddings
from modules.history_db import (
    DEFAULT_DB_PATH,
    HistoryStorageError,
    delete_chunk_embeddings_for_hash,
    get_chunk_embeddings,
    get_document_chunks_by_hash,
    save_chunk_embeddings,
)


@dataclass(frozen=True)
class HybridRankingConfig:
    """Configurable Reciprocal Rank Fusion (RRF) parameters."""

    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 1.2


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity safely between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _collect_doc_chunks(
    documents: list[dict[str, Any]],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Collect normalized chunk dicts for all documents from memory or SQLite."""
    collected: list[dict[str, Any]] = []
    for doc in documents or []:
        fname = str(doc.get("filename") or "document.pdf")
        fhash = str(doc.get("file_hash") or "").strip()
        chunks = list(doc.get("chunks") or [])
        if not chunks and fhash:
            try:
                chunks = get_document_chunks_by_hash(fhash, db_path=db_path)
            except HistoryStorageError:
                chunks = []
        if not chunks and doc.get("pages"):
            # Legacy in-memory doc without precomputed chunks
            for page in doc.get("pages", []):
                txt = str(page.get("text") or "").strip()
                if not txt:
                    continue
                pno = int(page.get("page_number") or 1)
                doc_key = fhash or hashlib.sha256(fname.encode("utf-8")).hexdigest()
                chunks.append(
                    {
                        "chunk_id": f"{doc_key}:p{pno}",
                        "file_hash": doc_key,
                        "filename": fname,
                        "page_number": pno,
                        "chunk_index": pno - 1,
                        "section_title": "",
                        "text": txt,
                    }
                )
        for ch in chunks:
            txt = str(ch.get("text") or "").strip()
            if not txt:
                continue
            doc_hash = str(ch.get("file_hash") or fhash or hashlib.sha256(fname.encode("utf-8")).hexdigest())
            cidx = int(ch.get("chunk_index") or 0)
            cid = str(ch.get("chunk_id") or f"{doc_hash}:{cidx}")
            collected.append(
                {
                    "chunk_id": cid,
                    "file_hash": doc_hash,
                    "filename": str(ch.get("filename") or fname),
                    "page_number": int(ch.get("page_number") or 1),
                    "chunk_index": cidx,
                    "section_title": str(ch.get("section_title") or ""),
                    "text": txt,
                }
            )
    return collected


def index_document_embeddings(
    documents: list[dict[str, Any]],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    batch_size: int = 32,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Incrementally generate and cache Gemini embeddings for document chunks in SQLite."""
    chunks = _collect_doc_chunks(documents, db_path=db_path)
    if not chunks:
        return {"indexed_count": 0, "cached_count": 0, "embedding_model": model}

    file_hashes = sorted({c["file_hash"] for c in chunks})
    if force_reindex:
        for fh in file_hashes:
            delete_chunk_embeddings_for_hash(fh, embedding_model=model, db_path=db_path)
        existing = {}
    else:
        existing = get_chunk_embeddings(file_hashes, embedding_model=model, db_path=db_path)

    missing = [c for c in chunks if c["chunk_id"] not in existing]
    if not missing:
        return {
            "indexed_count": 0,
            "cached_count": len(chunks),
            "embedding_model": model,
        }

    vectors, used_model = generate_embeddings(
        [c["text"] for c in missing],
        model=model,
        api_key=api_key,
        task_type="RETRIEVAL_DOCUMENT",
        batch_size=batch_size,
    )

    to_save = []
    for ch, vec in zip(missing, vectors):
        to_save.append(
            {
                "chunk_id": ch["chunk_id"],
                "file_hash": ch["file_hash"],
                "embedding_model": used_model,
                "embedding_dimensions": len(vec),
                "embedding_vector": vec,
            }
        )
    saved = save_chunk_embeddings(to_save, db_path=db_path)
    return {
        "indexed_count": saved,
        "cached_count": len(chunks) - len(missing),
        "embedding_model": used_model,
    }


def semantic_search_chunks(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 10,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    auto_index: bool = True,
) -> list[dict[str, Any]]:
    """Search document chunks using cosine similarity over Gemini embeddings."""
    clean_q = (query or "").strip()
    if not clean_q or not documents:
        return []

    chunks = _collect_doc_chunks(documents, db_path=db_path)
    if not chunks:
        return []

    effective_model = model
    if auto_index:
        idx_info = index_document_embeddings(
            documents=documents,
            model=model,
            api_key=api_key,
            db_path=db_path,
        )
        effective_model = idx_info.get("embedding_model", model)

    file_hashes = sorted({c["file_hash"] for c in chunks})
    stored = get_chunk_embeddings(file_hashes, embedding_model=effective_model, db_path=db_path)
    if not stored:
        return []

    query_vecs, _ = generate_embeddings(
        [clean_q],
        model=effective_model,
        api_key=api_key,
        task_type="RETRIEVAL_QUERY",
        batch_size=1,
    )
    if not query_vecs:
        return []
    q_vec = np.asarray(query_vecs[0], dtype=np.float32)
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0.0:
        return []
    q_unit = q_vec / q_norm

    candidate_chunks: list[dict[str, Any]] = []
    candidate_vecs: list[list[float]] = []
    candidate_models: list[str] = []
    q_dim = len(q_vec)
    for ch in chunks:
        emb = stored.get(ch["chunk_id"])
        if not emb or len(emb.get("vector") or []) != q_dim:
            continue
        candidate_chunks.append(ch)
        candidate_vecs.append(emb["vector"])
        candidate_models.append(emb["embedding_model"])

    if not candidate_chunks:
        return []

    mat = np.asarray(candidate_vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    sims = (mat @ q_unit) / safe_norms
    sims = np.where(norms == 0.0, 0.0, sims)

    scored: list[dict[str, Any]] = []
    for ch, emb_model, sim_val in zip(candidate_chunks, candidate_models, sims.tolist()):
        sim = float(sim_val)
        scored.append(
            {
                "chunk_id": ch["chunk_id"],
                "file_hash": ch["file_hash"],
                "filename": ch["filename"],
                "page_number": ch["page_number"],
                "chunk_index": ch["chunk_index"],
                "section_title": ch["section_title"],
                "text": ch["text"],
                "semantic_score": sim,
                "score": sim,
                "embedding_model": emb_model,
            }
        )

    scored.sort(key=lambda item: item["semantic_score"], reverse=True)
    return scored[: max(1, int(top_k))]


def _chunk_dedup_key(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])
    return (
        f"{chunk.get('filename', '')}::"
        f"{chunk.get('page_number', 0)}::"
        f"{chunk.get('chunk_index', '')}::"
        f"{str(chunk.get('text', ''))[:100]}"
    )


def reciprocal_rank_fusion(
    lexical_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 10,
    config: HybridRankingConfig | None = None,
) -> list[dict[str, Any]]:
    """Combine lexical and semantic ranked lists via configurable Reciprocal Rank Fusion (RRF)."""
    cfg = config or HybridRankingConfig()
    k = max(1, int(cfg.rrf_k))
    fused: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(lexical_results or [], start=1):
        key = _chunk_dedup_key(item)
        entry = fused.setdefault(key, dict(item))
        entry["lexical_score"] = float(item.get("score") or 0.0)
        entry["rrf_score"] = float(entry.get("rrf_score", 0.0)) + (cfg.lexical_weight / (k + rank))

    for rank, item in enumerate(semantic_results or [], start=1):
        key = _chunk_dedup_key(item)
        if key not in fused:
            entry = dict(item)
            fused[key] = entry
        else:
            entry = fused[key]
        entry["semantic_score"] = float(item.get("semantic_score") or item.get("score") or 0.0)
        if item.get("embedding_model"):
            entry["embedding_model"] = item["embedding_model"]
        entry["rrf_score"] = float(entry.get("rrf_score", 0.0)) + (cfg.semantic_weight / (k + rank))

    ranked = list(fused.values())
    for item in ranked:
        item["score"] = round(float(item.get("rrf_score", 0.0)), 6)
        item["retrieval_mode"] = "hybrid" if (lexical_results and semantic_results) else (
            "semantic" if semantic_results else "lexical"
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[: max(1, int(top_k))]
