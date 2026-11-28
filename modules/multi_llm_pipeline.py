"""Phase 2: Multi-LLM Parallel Processing and Master Orchestration pipeline."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from modules.ai_analysis import (
    AIServiceError,
    DEFAULT_MODEL,
    _sanitize_secret_text,
    generate_structured_json,
)
from modules.history_db import (
    DEFAULT_DB_PATH,
    HistoryStorageError,
    get_document_chunks_by_hash,
    load_worker_cache,
    save_worker_cache,
)
from modules.pdf_reader import chunk_document_pages

PROCESSING_VERSION = "v2.0"
MAX_CONCURRENT_WORKERS = 3
MAX_SECTION_BATCH_CHARS = 18000


class CitationRef(BaseModel):
    filename: str
    page_number: int


class WorkerFinding(BaseModel):
    finding: str
    page_numbers: list[int] = Field(default_factory=list)


class Worker1DocumentAnalysis(BaseModel):
    filename: str
    research_objectives: list[str]
    core_thesis: str
    methodology: str
    key_findings: list[WorkerFinding]
    limitations: list[str]
    key_terms: list[str]


class SectionSummaryItem(BaseModel):
    section_title: str
    page_numbers: list[int]
    summary: str
    key_points: list[str]


class Worker2SectionBatch(BaseModel):
    section_summaries: list[SectionSummaryItem]


class Worker2SectionSummary(BaseModel):
    filename: str
    section_summaries: list[SectionSummaryItem]
    document_summary: str


class MethodologyComparisonItem(BaseModel):
    aspect: str
    comparison: str
    documents: list[str]


class Worker3Concept(BaseModel):
    concept: str
    explanation: str
    documents: list[str]
    importance: Literal["High", "Medium", "Low"]


class Worker3Gap(BaseModel):
    title: str
    description: str
    affected_documents: list[str]
    severity: Literal["High", "Medium", "Low"]


class Worker3ResearchIntelligence(BaseModel):
    common_concepts: list[Worker3Concept]
    methodology_comparison: list[MethodologyComparisonItem]
    similarities: list[str]
    contradictions: list[str]
    research_gaps: list[Worker3Gap]
    cross_document_insights: str


class MasterSynthesizedFinding(BaseModel):
    claim: str
    citations: list[CitationRef]


class MasterDocumentDigest(BaseModel):
    filename: str
    summary: str
    key_pages: list[int] = Field(default_factory=list)


class MasterResearchReport(BaseModel):
    executive_summary: str
    document_digests: list[MasterDocumentDigest]
    synthesized_findings: list[MasterSynthesizedFinding]
    common_concepts: list[Worker3Concept]
    contradictions_and_gaps: list[str]
    open_research_questions: list[str]
    source_references: list[CitationRef]


@dataclass(frozen=True)
class WorkerModelConfig:
    """Independent Gemini model selection for each worker and the Master aggregator."""

    worker1_model: str = DEFAULT_MODEL
    worker2_model: str = "gemini-3.5-flash-lite"
    worker3_model: str = DEFAULT_MODEL
    master_model: str = DEFAULT_MODEL


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _merge_usage(target: dict[str, int], delta: dict[str, Any] | None) -> None:
    if not delta:
        return
    target["prompt_tokens"] += int(delta.get("prompt_tokens") or 0)
    target["output_tokens"] += int(delta.get("output_tokens") or 0)
    target["total_tokens"] += int(delta.get("total_tokens") or 0)


def _resolve_doc_chunks(
    doc: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return ordered section-aware chunks from memory, SQLite cache, or pages fallback."""
    if doc.get("chunks"):
        return list(doc["chunks"])
    file_hash = str(doc.get("file_hash") or "").strip()
    if file_hash:
        try:
            cached = get_document_chunks_by_hash(file_hash, db_path=db_path)
            if cached:
                return cached
        except HistoryStorageError:
            pass
    pages = doc.get("pages") or []
    if pages:
        return chunk_document_pages(
            pages=pages,
            filename=str(doc.get("filename") or "document.pdf"),
            file_hash=file_hash or hashlib.sha256(str(doc.get("filename", "")).encode("utf-8")).hexdigest(),
        )
    return []


def _group_chunks_into_batches(
    chunks: list[dict[str, Any]],
    max_batch_chars: int = MAX_SECTION_BATCH_CHARS,
) -> list[list[dict[str, Any]]]:
    """Group ordered chunks into bounded-size batches while keeping section boundaries clear."""
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for ch in chunks:
        text = str(ch.get("text") or "")
        cost = len(text) + 120
        if current_batch and (current_chars + cost > max_batch_chars):
            batches.append(current_batch)
            current_batch = [ch]
            current_chars = cost
        else:
            current_batch.append(ch)
            current_chars += cost

    if current_batch:
        batches.append(current_batch)
    return batches


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for ch in chunks:
        sec = ch.get("section_title") or "Section"
        pno = ch.get("page_number", 1)
        idx = ch.get("chunk_index", 0)
        blocks.append(f"[Chunk {idx} | Section: {sec} | Page {pno}]\n{ch.get('text', '')}")
    return "\n\n".join(blocks)


def _compute_corpus_hash(file_hashes: list[str]) -> str:
    joined = "|".join(sorted(h for h in file_hashes if h))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def run_worker1_document_analysis(
    doc: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
    processing_version: str = PROCESSING_VERSION,
    db_path: Path | str = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Worker 1: Analyze a single document for objectives, thesis, methodology, and findings."""
    filename = str(doc.get("filename") or "document.pdf")
    file_hash = str(doc.get("file_hash") or hashlib.sha256(filename.encode("utf-8")).hexdigest())

    if use_cache:
        try:
            cached = load_worker_cache(
                file_hash=file_hash,
                worker_name="worker1_analysis",
                processing_version=processing_version,
                completed_only=True,
                db_path=db_path,
            )
            if cached and cached.get("result"):
                out = dict(cached["result"])
                out["_from_cache"] = True
                out["_model_used"] = cached.get("model_used", model)
                out["_usage"] = _zero_usage()
                return out
        except HistoryStorageError:
            pass

    chunks = _resolve_doc_chunks(doc, db_path=db_path)
    if not chunks:
        raise AIServiceError(f"No readable chunks found for '{filename}'.")

    # Select balanced representation across start, middle, and end if single doc exceeds 90k chars
    formatted = _format_chunks_for_prompt(chunks)
    if len(formatted) > 90000:
        step = max(1, len(chunks) // 45)
        sampled = chunks[::step]
        formatted = _format_chunks_for_prompt(sampled)

    system_instruction = (
        "You are Worker 1 (Document Analysis Specialist). Extract the document's research objectives, "
        "core thesis, methodology, key findings with exact page numbers, limitations, and key terms. "
        "Stay strictly grounded in the provided chunks and never invent findings or page numbers."
    )
    prompt = (
        f"DOCUMENT FILENAME: {filename}\n"
        f"TITLE: {doc.get('title', filename)}\n\n"
        f"CHUNKS:\n{formatted}"
    )

    try:
        res = generate_structured_json(
            prompt=prompt,
            schema=Worker1DocumentAnalysis,
            system_instruction=system_instruction,
            model=model,
            api_key=api_key,
            temperature=0.15,
            allow_fallback=allow_fallback,
        )
        res["filename"] = filename
        res["file_hash"] = file_hash
        res["_from_cache"] = False
        try:
            save_worker_cache(
                file_hash=file_hash,
                worker_name="worker1_analysis",
                processing_version=processing_version,
                model_used=res.get("_model_used", model),
                status="completed",
                result_data=res,
                token_usage=res.get("_usage"),
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
        return res
    except Exception as exc:
        safe_err = _sanitize_secret_text(str(exc), api_key)
        try:
            save_worker_cache(
                file_hash=file_hash,
                worker_name="worker1_analysis",
                processing_version=processing_version,
                model_used=model,
                status="failed",
                result_data=None,
                error_message=safe_err,
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
        raise


def run_worker2_section_summarization(
    doc: dict[str, Any],
    model: str = "gemini-3.5-flash-lite",
    api_key: str | None = None,
    allow_fallback: bool = True,
    processing_version: str = PROCESSING_VERSION,
    db_path: Path | str = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Worker 2: Summarize stored PDF chunks section-by-section, preserving pages & filenames."""
    filename = str(doc.get("filename") or "document.pdf")
    file_hash = str(doc.get("file_hash") or hashlib.sha256(filename.encode("utf-8")).hexdigest())

    if use_cache:
        try:
            cached = load_worker_cache(
                file_hash=file_hash,
                worker_name="worker2_sections",
                processing_version=processing_version,
                completed_only=True,
                db_path=db_path,
            )
            if cached and cached.get("result"):
                out = dict(cached["result"])
                out["_from_cache"] = True
                out["_model_used"] = cached.get("model_used", model)
                out["_usage"] = _zero_usage()
                return out
        except HistoryStorageError:
            pass

    chunks = _resolve_doc_chunks(doc, db_path=db_path)
    if not chunks:
        raise AIServiceError(f"No readable chunks found for '{filename}'.")

    batches = _group_chunks_into_batches(chunks)
    all_sections: list[dict[str, Any]] = []
    total_usage = _zero_usage()
    models_used: list[str] = []

    system_instruction = (
        "You are Worker 2 (Section Summarization Specialist). Summarize each section in the provided "
        "PDF chunks faithfully, preserving exact section titles and page numbers. Never fabricate claims."
    )

    try:
        for batch_idx, batch in enumerate(batches, start=1):
            batch_prompt = (
                f"DOCUMENT FILENAME: {filename}\n"
                f"BATCH {batch_idx} OF {len(batches)}\n\n"
                f"CHUNKS:\n{_format_chunks_for_prompt(batch)}"
            )
            batch_res = generate_structured_json(
                prompt=batch_prompt,
                schema=Worker2SectionBatch,
                system_instruction=system_instruction,
                model=model,
                api_key=api_key,
                temperature=0.15,
                allow_fallback=allow_fallback,
            )
            _merge_usage(total_usage, batch_res.get("_usage"))
            used_m = batch_res.get("_model_used", model)
            if used_m not in models_used:
                models_used.append(used_m)
            for sec in batch_res.get("section_summaries", []):
                all_sections.append(sec)

        combined_summary = " ".join(
            f"[{s.get('section_title', 'Section')} (pp. {', '.join(str(p) for p in s.get('page_numbers', []))})]: {s.get('summary', '')}"
            for s in all_sections
        ).strip()

        result = {
            "filename": filename,
            "file_hash": file_hash,
            "section_summaries": all_sections,
            "document_summary": combined_summary,
            "_model_used": models_used[-1] if models_used else model,
            "_usage": total_usage,
            "_from_cache": False,
        }
        try:
            save_worker_cache(
                file_hash=file_hash,
                worker_name="worker2_sections",
                processing_version=processing_version,
                model_used=result["_model_used"],
                status="completed",
                result_data=result,
                token_usage=total_usage,
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
        return result
    except Exception as exc:
        safe_err = _sanitize_secret_text(str(exc), api_key)
        try:
            save_worker_cache(
                file_hash=file_hash,
                worker_name="worker2_sections",
                processing_version=processing_version,
                model_used=model,
                status="failed",
                result_data=None,
                error_message=safe_err,
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
        raise


def run_worker3_research_intelligence(
    worker1_results: list[dict[str, Any]],
    worker2_results: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
    processing_version: str = PROCESSING_VERSION,
    db_path: Path | str = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Worker 3: Cross-document concept extraction, comparison, contradictions, and research gaps."""
    if not worker1_results and not worker2_results:
        raise AIServiceError("Worker 3 requires at least one completed document analysis.")

    corpus_hash = _compute_corpus_hash(
        [str(r.get("file_hash") or "") for r in (worker1_results or worker2_results)]
    )
    if use_cache and corpus_hash:
        try:
            cached = load_worker_cache(
                file_hash=corpus_hash,
                worker_name="worker3_intelligence",
                processing_version=processing_version,
                completed_only=True,
                db_path=db_path,
            )
            if cached and cached.get("result"):
                out = dict(cached["result"])
                out["_from_cache"] = True
                out["_model_used"] = cached.get("model_used", model)
                out["_usage"] = _zero_usage()
                return out
        except HistoryStorageError:
            pass

    compact_w1 = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in worker1_results
    ]
    compact_w2 = [
        {
            "filename": r.get("filename"),
            "document_summary": r.get("document_summary", ""),
            "section_count": len(r.get("section_summaries", [])),
        }
        for r in worker2_results
    ]

    system_instruction = (
        "You are Worker 3 (Research Intelligence & Comparison Specialist). Compare methodologies, "
        "identify common concepts, highlight similarities, contradictions, and research gaps across "
        "the documents using only the provided Worker 1 and Worker 2 outputs."
    )
    prompt = (
        f"WORKER 1 DOCUMENT ANALYSES:\n{json.dumps(compact_w1, indent=2)}\n\n"
        f"WORKER 2 SECTION SUMMARIES:\n{json.dumps(compact_w2, indent=2)}"
    )

    res = generate_structured_json(
        prompt=prompt,
        schema=Worker3ResearchIntelligence,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.2,
        allow_fallback=allow_fallback,
    )
    res["_from_cache"] = False
    if corpus_hash:
        try:
            save_worker_cache(
                file_hash=corpus_hash,
                worker_name="worker3_intelligence",
                processing_version=processing_version,
                model_used=res.get("_model_used", model),
                status="completed",
                result_data=res,
                token_usage=res.get("_usage"),
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
    return res


def run_master_aggregator(
    worker1_results: list[dict[str, Any]],
    worker2_results: list[dict[str, Any]],
    worker3_result: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
    processing_version: str = PROCESSING_VERSION,
    db_path: Path | str = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Master LLM: Synthesize Worker 1-3 outputs into a deduplicated, citation-grounded final report."""
    corpus_hash = _compute_corpus_hash(
        [str(r.get("file_hash") or "") for r in (worker1_results or worker2_results)]
    )
    if use_cache and corpus_hash:
        try:
            cached = load_worker_cache(
                file_hash=corpus_hash,
                worker_name="master_report",
                processing_version=processing_version,
                completed_only=True,
                db_path=db_path,
            )
            if cached and cached.get("result"):
                out = dict(cached["result"])
                out["_from_cache"] = True
                out["_model_used"] = cached.get("model_used", model)
                out["_usage"] = _zero_usage()
                return out
        except HistoryStorageError:
            pass

    clean_w1 = [{k: v for k, v in r.items() if not k.startswith("_")} for r in worker1_results]
    clean_w2 = [{k: v for k, v in r.items() if not k.startswith("_")} for r in worker2_results]
    clean_w3 = {k: v for k, v in (worker3_result or {}).items() if not k.startswith("_")}

    system_instruction = (
        "You are the Master Research Aggregator LLM. Combine Worker 1, Worker 2, and Worker 3 outputs "
        "into a comprehensive, deduplicated final research report. Preserve exact PDF filenames and "
        "page_number citations from the worker outputs. NEVER invent findings, filenames, or page numbers."
    )
    prompt = (
        f"WORKER 1 ANALYSES:\n{json.dumps(clean_w1, indent=2)}\n\n"
        f"WORKER 2 SECTION SUMMARIES:\n{json.dumps(clean_w2, indent=2)}\n\n"
        f"WORKER 3 RESEARCH INTELLIGENCE:\n{json.dumps(clean_w3, indent=2)}"
    )

    res = generate_structured_json(
        prompt=prompt,
        schema=MasterResearchReport,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.15,
        allow_fallback=allow_fallback,
    )
    res["_from_cache"] = False
    if corpus_hash:
        try:
            save_worker_cache(
                file_hash=corpus_hash,
                worker_name="master_report",
                processing_version=processing_version,
                model_used=res.get("_model_used", model),
                status="completed",
                result_data=res,
                token_usage=res.get("_usage"),
                db_path=db_path,
            )
        except HistoryStorageError:
            pass
    return res


def run_multi_llm_pipeline(
    documents: list[dict[str, Any]],
    model_config: WorkerModelConfig | None = None,
    api_key: str | None = None,
    allow_fallback: bool = True,
    max_workers: int = MAX_CONCURRENT_WORKERS,
    processing_version: str = PROCESSING_VERSION,
    db_path: Path | str = DEFAULT_DB_PATH,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute bounded parallel Worker 1 & Worker 2 tasks, then Worker 3 and Master Aggregator."""
    if not documents:
        raise AIServiceError("Upload at least one PDF document before running the Multi-LLM pipeline.")

    cfg = model_config or WorkerModelConfig()
    bounded_workers = min(MAX_CONCURRENT_WORKERS, max(1, int(max_workers)))

    worker1_by_hash: dict[str, dict[str, Any]] = {}
    worker2_by_hash: dict[str, dict[str, Any]] = {}
    failed_documents: list[dict[str, str]] = []
    total_usage = _zero_usage()
    cache_hits = 0

    def _run_doc_w1(d: dict[str, Any]) -> dict[str, Any]:
        return run_worker1_document_analysis(
            doc=d,
            model=cfg.worker1_model,
            api_key=api_key,
            allow_fallback=allow_fallback,
            processing_version=processing_version,
            db_path=db_path,
            use_cache=use_cache,
        )

    def _run_doc_w2(d: dict[str, Any]) -> dict[str, Any]:
        return run_worker2_section_summarization(
            doc=d,
            model=cfg.worker2_model,
            api_key=api_key,
            allow_fallback=allow_fallback,
            processing_version=processing_version,
            db_path=db_path,
            use_cache=use_cache,
        )

    with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        future_map = {}
        for doc in documents:
            fhash = str(doc.get("file_hash") or doc.get("filename") or "")
            fname = str(doc.get("filename") or "document.pdf")
            future_map[executor.submit(_run_doc_w1, doc)] = ("worker1", fhash, fname)
            future_map[executor.submit(_run_doc_w2, doc)] = ("worker2", fhash, fname)

        for fut in as_completed(future_map):
            role, fhash, fname = future_map[fut]
            try:
                res = fut.result()
                _merge_usage(total_usage, res.get("_usage"))
                if res.get("_from_cache"):
                    cache_hits += 1
                if role == "worker1":
                    worker1_by_hash[fhash] = res
                else:
                    worker2_by_hash[fhash] = res
            except Exception as exc:
                failed_documents.append(
                    {
                        "filename": fname,
                        "file_hash": fhash,
                        "worker": role,
                        "error": _sanitize_secret_text(str(exc), api_key),
                    }
                )

    worker1_results = [
        worker1_by_hash[str(d.get("file_hash") or d.get("filename") or "")]
        for d in documents
        if str(d.get("file_hash") or d.get("filename") or "") in worker1_by_hash
    ]
    worker2_results = [
        worker2_by_hash[str(d.get("file_hash") or d.get("filename") or "")]
        for d in documents
        if str(d.get("file_hash") or d.get("filename") or "") in worker2_by_hash
    ]

    if not worker1_results and not worker2_results:
        return {
            "status": "failed",
            "worker1_results": [],
            "worker2_results": [],
            "worker3_result": None,
            "master_report": None,
            "failed_documents": failed_documents,
            "cache_hits": cache_hits,
            "token_usage": total_usage,
        }

    worker3_result = run_worker3_research_intelligence(
        worker1_results=worker1_results,
        worker2_results=worker2_results,
        model=cfg.worker3_model,
        api_key=api_key,
        allow_fallback=allow_fallback,
        processing_version=processing_version,
        db_path=db_path,
        use_cache=use_cache,
    )
    _merge_usage(total_usage, worker3_result.get("_usage"))
    if worker3_result.get("_from_cache"):
        cache_hits += 1

    master_report = run_master_aggregator(
        worker1_results=worker1_results,
        worker2_results=worker2_results,
        worker3_result=worker3_result,
        model=cfg.master_model,
        api_key=api_key,
        allow_fallback=allow_fallback,
        processing_version=processing_version,
        db_path=db_path,
        use_cache=use_cache,
    )
    _merge_usage(total_usage, master_report.get("_usage"))
    if master_report.get("_from_cache"):
        cache_hits += 1

    status = "partial" if failed_documents else "completed"
    return {
        "status": status,
        "worker1_results": worker1_results,
        "worker2_results": worker2_results,
        "worker3_result": worker3_result,
        "master_report": master_report,
        "failed_documents": failed_documents,
        "cache_hits": cache_hits,
        "token_usage": total_usage,
    }
