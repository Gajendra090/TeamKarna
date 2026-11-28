"""Gemini AI integration for structured knowledge analysis, common concepts, and PDF Q&A."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from modules.pdf_reader import build_corpus_context, retrieve_relevant_chunks

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"
RECOMMENDED_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-4-hydrogen",
]
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-4-hydrogen",
]
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
FALLBACK_EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "your_api_key_here"}
REQUEST_TIMEOUT_MS = 90_000
_DAILY_EXHAUSTED_MODELS: set[str] = set()
_MODEL_STATE_LOCK = threading.Lock()


def _get_daily_exhausted_snapshot() -> set[str]:
    with _MODEL_STATE_LOCK:
        return set(_DAILY_EXHAUSTED_MODELS)


def _mark_model_daily_exhausted(model_name: str) -> None:
    with _MODEL_STATE_LOCK:
        _DAILY_EXHAUSTED_MODELS.add(model_name)


def clear_daily_exhausted_models() -> None:
    """Reset the in-memory daily-exhausted model set safely."""
    with _MODEL_STATE_LOCK:
        _DAILY_EXHAUSTED_MODELS.clear()


class AIServiceError(Exception):
    """Raised when Gemini configuration, API calls, or parsing fail."""


class DocumentBreakdown(BaseModel):
    filename: str = Field(description="Exact PDF filename.")
    core_thesis: str = Field(description="Main objective, thesis, or central claim.")
    methodology: str = Field(description="Methods, models, datasets, or experimental approach.")
    key_findings: list[str] = Field(description="3 to 5 concrete findings or contributions.")
    limitations: list[str] = Field(description="Explicit or apparent limitations in the paper.")
    key_terms: list[str] = Field(description="Key concepts or technical keywords from this paper.")


class CommonConcept(BaseModel):
    concept: str = Field(description="Shared theme, technique, or concept.")
    explanation: str = Field(description="How this concept connects or varies across documents.")
    documents: list[str] = Field(description="Filenames where this concept appears.")
    importance: Literal["High", "Medium", "Low"]


class GlossaryEntry(BaseModel):
    term: str
    definition: str = Field(description="Clear, concise definition for rapid understanding.")
    source_document: str


class KnowledgeAnalysisResult(BaseModel):
    corpus_overview: str = Field(
        description="High-signal executive synthesis of the uploaded documents."
    )
    key_takeaways: list[str] = Field(
        description="3 to 6 rapid-internalization takeaways across the corpus."
    )
    document_analyses: list[DocumentBreakdown]
    common_concepts: list[CommonConcept] = Field(
        description="Concepts shared across multiple documents (or recurring themes if 1 document)."
    )
    terminology_glossary: list[GlossaryEntry]


class CitationItem(BaseModel):
    filename: str
    page_number: int
    evidence: str = Field(description="Brief supporting detail or paraphrase from that page.")


class ResearchAnswer(BaseModel):
    answer: str = Field(description="Clear markdown answer strictly grounded in the PDF excerpts.")
    citations: list[CitationItem]
    confidence: Literal["High", "Medium", "Low"]
    follow_up_questions: list[str]


def _sanitize_secret_text(text: str, secret: str | None = None) -> str:
    """Redact any API key material before displaying messages to users."""
    cleaned = text or ""
    if secret:
        cleaned = cleaned.replace(secret, "[REDACTED_KEY]")
    cleaned = re.sub(r"AIza[0-9A-Za-z\-_]{16,}", "[REDACTED_KEY]", cleaned)
    return cleaned


def load_env_config(env_path: Path | str = ".env") -> dict[str, str]:
    """Load environment variables from .env if present without crashing if absent."""
    path = Path(env_path)
    if path.exists():
        load_dotenv(dotenv_path=path, override=True)
    else:
        load_dotenv(override=False)

    raw_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if raw_key in PLACEHOLDER_KEYS:
        raw_key = ""

    raw_model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip().strip('"').strip("'") or DEFAULT_MODEL
    if raw_model.startswith("models/"):
        raw_model = raw_model[len("models/") :]

    return {
        "api_key": raw_key,
        "model": raw_model,
        "env_file_exists": "true" if path.exists() else "false",
    }


def resolve_api_key(override_key: str | None = None) -> str:
    """Resolve Gemini API key preferring session override if provided, else .env."""
    if override_key:
        cleaned = override_key.strip().strip('"').strip("'")
        if cleaned and cleaned not in PLACEHOLDER_KEYS:
            return cleaned
    config = load_env_config()
    return config["api_key"]


def is_api_configured(override_key: str | None = None) -> tuple[bool, str]:
    """Return configuration status and human-readable source note."""
    if override_key:
        cleaned = override_key.strip().strip('"').strip("'")
        if cleaned and cleaned not in PLACEHOLDER_KEYS:
            return True, "Active (session key)"
    config = load_env_config()
    if config["api_key"]:
        source = ".env file" if config["env_file_exists"] == "true" else "environment variable"
        return True, f"Active ({source})"
    return False, "Missing GEMINI_API_KEY in .env"


def get_gemini_client(
    api_key: str | None = None,
    retry_attempts: int = 2,
) -> genai.Client:
    """Instantiate a Google GenAI client with explicit HTTP timeout and bounded retries."""
    resolved = resolve_api_key(api_key)
    if not resolved:
        raise AIServiceError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to your .env file "
            "or enter it in the sidebar configuration."
        )
    try:
        http_opts = types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=max(1, retry_attempts),
                initial_delay=0.8,
                max_delay=2.5,
                exp_base=2.0,
            ),
        )
        return genai.Client(api_key=resolved, http_options=http_opts)
    except Exception as exc:
        safe_msg = _sanitize_secret_text(str(exc), resolved)
        raise AIServiceError(f"Failed to initialize Gemini client: {safe_msg}") from exc


def _is_auth_error(exc: Exception) -> bool:
    if not isinstance(exc, errors.APIError):
        return False
    code = getattr(exc, "code", None)
    status = (getattr(exc, "status", "") or "").upper()
    detail = (getattr(exc, "message", "") or str(exc)).upper()
    return (
        code in (401, 403)
        or "API_KEY_INVALID" in detail
        or "API KEY NOT VALID" in detail
        or "PERMISSION_DENIED" in status
    )


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    if not isinstance(exc, errors.APIError):
        return False
    code = getattr(exc, "code", None)
    detail = (getattr(exc, "message", "") or str(exc)).upper()
    return code == 429 and ("PER DAY" in detail or "LIMIT: 0" in detail)


def _is_retryable_model_error(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        status = (getattr(exc, "status", "") or "").upper()
        return code in (404, 429, 500, 502, 503, 504) or status in {
            "NOT_FOUND",
            "RESOURCE_EXHAUSTED",
            "INTERNAL",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        }
    return False


def _format_api_error(
    exc: Exception,
    model: str,
    api_key_for_redaction: str | None = None,
    attempted_models: list[str] | None = None,
) -> AIServiceError:
    """Translate Gemini SDK/HTTP errors into clear, diagnostic, actionable messages."""
    if isinstance(exc, AIServiceError):
        return exc

    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        status = (getattr(exc, "status", "") or "").strip()
        raw_detail = (getattr(exc, "message", None) or str(exc)).strip()
        detail = _sanitize_secret_text(raw_detail, api_key_for_redaction)
        first_line = detail.splitlines()[0] if detail else "Unknown error"
        upper_detail = detail.upper()
        upper_status = status.upper()

        if (
            code in (401, 403)
            or "API_KEY" in upper_detail
            or "API KEY NOT VALID" in upper_detail
            or "PERMISSION" in upper_status
        ):
            return AIServiceError(
                f"Gemini authentication failed (HTTP {code or 401} {status}): {first_line} "
                "Verify that GEMINI_API_KEY in .env is valid, active, and has Gemini API access."
            )

        if code == 404 or "NOT_FOUND" in upper_status:
            return AIServiceError(
                f"Model '{model}' is unsupported or unavailable (HTTP 404 NOT_FOUND): {first_line} "
                "Select a supported model such as 'gemini-3.6-flash' or 'gemini-3.5-flash' in the sidebar."
            )

        if code == 429 or "RESOURCE_EXHAUSTED" in upper_status:
            if "LIMIT: 0" in upper_detail:
                return AIServiceError(
                    f"Quota unavailable for '{model}' (HTTP 429 RESOURCE_EXHAUSTED): "
                    "Your current API tier has a limit of 0 for this model. "
                    "Switch to 'gemini-3.6-flash' or 'gemini-3.5-flash' in the sidebar, or enable billing."
                )
            retry_match = re.search(r"retry in ([\w\.]+)", detail, re.IGNORECASE)
            retry_hint = f" Suggested wait: {retry_match.group(1)}." if retry_match else ""
            return AIServiceError(
                f"Rate limit or quota exceeded for '{model}' (HTTP 429 RESOURCE_EXHAUSTED).{retry_hint} "
                f"Details: {first_line}"
            )

        if code in (500, 502, 503, 504) or upper_status in {"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}:
            tried_note = (
                f" (tried: {', '.join(attempted_models)})"
                if attempted_models and len(attempted_models) > 1
                else ""
            )
            return AIServiceError(
                f"Gemini model '{model}' returned HTTP {code or 503} ({status or 'UNAVAILABLE'}){tried_note}: "
                f"{first_line} Try selecting 'gemini-3.6-flash' or 'gemini-3.5-flash' in the sidebar."
            )

        return AIServiceError(
            f"Gemini API error on '{model}' (HTTP {code or 'unknown'} {status}): {first_line}"
        )

    msg = _sanitize_secret_text(str(exc), api_key_for_redaction)
    lower_msg = msg.lower()
    if any(tok in lower_msg for tok in ("timeout", "timed out", "connect", "dns", "network", "ssl")):
        return AIServiceError(
            f"Network or timeout error reaching Gemini API for '{model}': {msg}. "
            "Check your internet connection and retry."
        )

    return AIServiceError(f"AI request failed on '{model}': {msg}")


def _extract_json_payload(raw_text: str) -> Any:
    """Parse JSON safely even if wrapped in markdown fences."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def test_gemini_connection(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a lightweight connectivity and model availability check without exposing secrets."""
    resolved = resolve_api_key(api_key)
    selected_model = (model or DEFAULT_MODEL).strip()
    if selected_model.startswith("models/"):
        selected_model = selected_model[len("models/") :]

    client = get_gemini_client(resolved, retry_attempts=1)

    try:
        available = [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        ]
    except Exception as exc:
        raise _format_api_error(exc, selected_model, api_key_for_redaction=resolved) from exc

    try:
        response = client.models.generate_content(
            model=selected_model,
            contents="Reply with OK.",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=16),
        )
        reply = (getattr(response, "text", "") or "").strip()
        return {
            "ok": True,
            "model": selected_model,
            "reply": reply or "OK",
            "available_count": len(available),
            "available_models": available,
        }
    except Exception as exc:
        raise _format_api_error(exc, selected_model, api_key_for_redaction=resolved) from exc


def _extract_usage_metadata(response: Any) -> dict[str, int]:
    """Extract token usage counts from a Gemini response when available."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        prompt_tok = int(usage.get("prompt_token_count") or 0)
        cand_tok = int(usage.get("candidates_token_count") or 0)
        total_tok = int(usage.get("total_token_count") or (prompt_tok + cand_tok))
    else:
        prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        cand_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tok = int(getattr(usage, "total_token_count", 0) or (prompt_tok + cand_tok))
    return {
        "prompt_tokens": prompt_tok,
        "output_tokens": cand_tok,
        "total_tokens": total_tok,
    }


def generate_structured_json(
    prompt: str | list[Any],
    schema: Type[T],
    system_instruction: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.2,
    allow_fallback: bool = True,
    max_rate_limit_retries: int = 2,
    retry_backoff_base: float = 0.8,
) -> dict[str, Any]:
    """Call Gemini and validate the JSON response against a Pydantic schema."""
    resolved_key = resolve_api_key(api_key)
    client = get_gemini_client(resolved_key, retry_attempts=2)

    primary_model = (model or DEFAULT_MODEL).strip()
    if primary_model.startswith("models/"):
        primary_model = primary_model[len("models/") :]

    exhausted = _get_daily_exhausted_snapshot()
    candidates = [primary_model]
    if allow_fallback:
        for fallback in FALLBACK_MODELS:
            if fallback not in candidates and fallback not in exhausted:
                candidates.append(fallback)
        # Skip primary up front only if another non-exhausted fallback exists
        if primary_model in exhausted and len(candidates) > 1:
            candidates = [c for c in candidates if c != primary_model] + [primary_model]

    last_exc: Exception | None = None
    last_model = primary_model
    attempted: list[str] = []

    for candidate in candidates:
        attempted.append(candidate)
        last_model = candidate
        response = None
        for rl_attempt in range(max(0, max_rate_limit_retries) + 1):
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                break
            except Exception as exc:
                last_exc = exc
                if _is_daily_quota_exhausted(exc):
                    _mark_model_daily_exhausted(candidate)
                    break
                if _is_auth_error(exc):
                    raise _format_api_error(
                        exc, candidate, api_key_for_redaction=resolved_key, attempted_models=attempted
                    ) from exc
                is_429 = isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 429
                if is_429 and rl_attempt < max(0, max_rate_limit_retries):
                    if retry_backoff_base > 0:
                        delay = min(3.0, retry_backoff_base * (2 ** rl_attempt) + random.uniform(0.05, 0.25))
                        time.sleep(delay)
                    continue
                break

        if response is None:
            assert last_exc is not None
            if allow_fallback and _is_retryable_model_error(last_exc):
                continue
            raise _format_api_error(
                last_exc, candidate, api_key_for_redaction=resolved_key, attempted_models=attempted
            ) from last_exc

        parsed_result: dict[str, Any] | None = None
        if getattr(response, "parsed", None) is not None:
            parsed_obj = response.parsed
            if isinstance(parsed_obj, BaseModel):
                parsed_result = parsed_obj.model_dump()
            elif isinstance(parsed_obj, dict):
                parsed_result = schema.model_validate(parsed_obj).model_dump()

        if parsed_result is None:
            raw_text = getattr(response, "text", "") or ""
            if not raw_text.strip():
                last_exc = AIServiceError(
                    f"Gemini model '{candidate}' returned an empty response."
                )
                if allow_fallback:
                    continue
                raise last_exc
            try:
                payload = _extract_json_payload(raw_text)
                parsed_result = schema.model_validate(payload).model_dump()
            except Exception as exc:
                raise AIServiceError(
                    f"Could not parse structured JSON response from '{candidate}': {exc}"
                ) from exc

        parsed_result["_model_used"] = candidate
        parsed_result["_usage"] = _extract_usage_metadata(response)
        if candidate != primary_model:
            parsed_result["_fallback_note"] = (
                f"Primary model '{primary_model}' was unavailable; completed with '{candidate}'."
            )
        return parsed_result

    assert last_exc is not None
    raise _format_api_error(
        last_exc,
        last_model,
        api_key_for_redaction=resolved_key,
        attempted_models=attempted,
    ) from last_exc


def generate_embeddings(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    allow_fallback: bool = True,
    max_rate_limit_retries: int = 2,
    retry_backoff_base: float = 0.8,
    batch_size: int = 32,
) -> tuple[list[list[float]], str]:
    """Generate batch embeddings via Gemini SDK with retries, rate-limit backoff, and fallback."""
    clean_texts = [str(t or "").strip() or " " for t in (texts or [])]
    if not clean_texts:
        return [], (model or DEFAULT_EMBEDDING_MODEL)

    resolved_key = resolve_api_key(api_key)
    client = get_gemini_client(resolved_key, retry_attempts=2)

    primary_model = (model or DEFAULT_EMBEDDING_MODEL).strip()
    if primary_model.startswith("models/"):
        primary_model = primary_model[len("models/") :]

    exhausted = _get_daily_exhausted_snapshot()
    candidates = [primary_model]
    if allow_fallback:
        for fb in FALLBACK_EMBEDDING_MODELS:
            if fb not in candidates and fb not in exhausted:
                candidates.append(fb)
        if primary_model in exhausted and len(candidates) > 1:
            candidates = [c for c in candidates if c != primary_model] + [primary_model]

    step = max(1, int(batch_size))
    last_exc: Exception | None = None
    last_model = primary_model
    attempted: list[str] = []

    for candidate in candidates:
        attempted.append(candidate)
        last_model = candidate
        all_vectors: list[list[float]] = []
        failed_candidate = False

        for start in range(0, len(clean_texts), step):
            batch = clean_texts[start : start + step]
            resp = None
            for rl_attempt in range(max(0, max_rate_limit_retries) + 1):
                try:
                    resp = client.models.embed_content(
                        model=candidate,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_daily_quota_exhausted(exc):
                        _mark_model_daily_exhausted(candidate)
                        break
                    if _is_auth_error(exc):
                        raise _format_api_error(
                            exc, candidate, api_key_for_redaction=resolved_key, attempted_models=attempted
                        ) from exc
                    is_429 = isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 429
                    if is_429 and rl_attempt < max(0, max_rate_limit_retries):
                        if retry_backoff_base > 0:
                            delay = min(3.0, retry_backoff_base * (2 ** rl_attempt) + random.uniform(0.05, 0.25))
                            time.sleep(delay)
                        continue
                    break

            if resp is None:
                failed_candidate = True
                break

            emb_items = getattr(resp, "embeddings", None) or []
            if len(emb_items) != len(batch):
                last_exc = AIServiceError(
                    f"Embedding count mismatch from '{candidate}': expected {len(batch)}, got {len(emb_items)}."
                )
                failed_candidate = True
                break

            for item in emb_items:
                vals = getattr(item, "values", None)
                if not vals:
                    last_exc = AIServiceError(f"Empty embedding vector returned by '{candidate}'.")
                    failed_candidate = True
                    break
                all_vectors.append([float(v) for v in vals])
            if failed_candidate:
                break

        if not failed_candidate and len(all_vectors) == len(clean_texts):
            return all_vectors, candidate

        if not (allow_fallback and last_exc is not None and _is_retryable_model_error(last_exc)):
            assert last_exc is not None
            raise _format_api_error(
                last_exc, candidate, api_key_for_redaction=resolved_key, attempted_models=attempted
            ) from last_exc

    assert last_exc is not None
    raise _format_api_error(
        last_exc, last_model, api_key_for_redaction=resolved_key, attempted_models=attempted
    ) from last_exc


def analyze_documents(
    documents: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Generate structured knowledge analysis and cross-document common concepts."""
    if not documents:
        raise AIServiceError("Upload at least one PDF document before running analysis.")

    corpus_text = build_corpus_context(documents, max_chars=120000)
    filenames = [doc["filename"] for doc in documents]

    system_instruction = (
        "You are KNOWLEDGE X, an expert research synthesis assistant designed for rapid "
        "knowledge internalization. Be concise, accurate, and strictly grounded in the "
        "provided research documents. Never fabricate findings, metrics, or citations."
    )

    prompt = (
        f"Analyze the following {len(documents)} research PDF(s): {', '.join(filenames)}.\n\n"
        "Tasks:\n"
        "1. Write an executive corpus_overview synthesizing what these documents establish.\n"
        "2. List 3-6 high-impact key_takeaways for rapid understanding.\n"
        "3. Provide a structured breakdown for EVERY uploaded filename in document_analyses "
        "(using the exact filename string).\n"
        "4. Identify common_concepts shared across the documents (or major recurring concepts "
        "if only one document is present), listing the exact filenames where each appears.\n"
        "5. Build a concise terminology_glossary of critical technical terms.\n\n"
        f"DOCUMENTS:\n{corpus_text}"
    )

    return generate_structured_json(
        prompt=prompt,
        schema=KnowledgeAnalysisResult,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.2,
        allow_fallback=allow_fallback,
    )


def answer_research_question(
    question: str,
    documents: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
    use_semantic: bool = True,
    embedding_model: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Answer a user question grounded strictly in uploaded PDF chunks."""
    clean_question = (question or "").strip()
    if not clean_question:
        raise AIServiceError("Please enter a question.")
    if not documents:
        raise AIServiceError("Upload at least one PDF before asking questions.")

    retrieve_kwargs: dict[str, Any] = {
        "top_k": 10,
        "use_semantic": use_semantic,
        "embedding_model": embedding_model,
        "api_key": api_key,
    }
    if db_path is not None:
        retrieve_kwargs["db_path"] = db_path

    relevant_chunks = retrieve_relevant_chunks(documents, clean_question, **retrieve_kwargs)
    if not relevant_chunks:
        raise AIServiceError("No readable text chunks found in the selected documents.")

    modes = {str(c.get("retrieval_mode") or "lexical") for c in relevant_chunks}
    if "hybrid" in modes:
        retrieval_mode = "hybrid"
    elif "semantic" in modes:
        retrieval_mode = "semantic"
    else:
        retrieval_mode = "lexical"

    excerpts = []
    for idx, chunk in enumerate(relevant_chunks, start=1):
        sec = f" | Section: {chunk['section_title']}" if chunk.get("section_title") else ""
        excerpts.append(
            f"[Excerpt {idx} | File: {chunk['filename']} | Page: {chunk['page_number']}{sec}]\n"
            f"{chunk['text']}"
        )
    context_block = "\n\n".join(excerpts)

    doc_summaries = "\n".join(
        f"- {d['filename']} (Title: {d.get('title', d['filename'])}, Pages: {d.get('page_count', 0)})"
        for d in documents
    )

    history_lines: list[str] = []
    for msg in (chat_history or [])[-6:]:
        role = msg.get("role", "user").upper()
        content = (msg.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content[:700]}")
    history_block = "\n".join(history_lines) if history_lines else "None"

    system_instruction = (
        "You are the KNOWLEDGE X Research Assistant. Answer questions strictly based on the "
        "provided PDF excerpts. Cite exact filenames and page numbers in citations. "
        "If the excerpts do not contain enough evidence to answer fully, state clearly what is "
        "supported and what is missing rather than guessing."
    )

    prompt = (
        f"LOADED DOCUMENTS:\n{doc_summaries}\n\n"
        f"RECENT CONVERSATION:\n{history_block}\n\n"
        f"RELEVANT PDF EXCERPTS:\n{context_block}\n\n"
        f"USER QUESTION:\n{clean_question}"
    )

    result = generate_structured_json(
        prompt=prompt,
        schema=ResearchAnswer,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.15,
        allow_fallback=allow_fallback,
    )
    result["_retrieval_mode"] = retrieval_mode
    result["_retrieved_chunks"] = [
        {
            "filename": c.get("filename", ""),
            "page_number": c.get("page_number", 1),
            "section_title": c.get("section_title", ""),
        }
        for c in relevant_chunks
    ]
    return result


def format_analysis_markdown(
    analysis: dict[str, Any],
    gaps: dict[str, Any] | None = None,
) -> str:
    """Export structured knowledge analysis and optional gap report as Markdown."""
    lines: list[str] = ["# KNOWLEDGE X — Research Synthesis Report", ""]

    if analysis:
        lines.extend(["## Corpus Overview", analysis.get("corpus_overview", ""), ""])
        if analysis.get("key_takeaways"):
            lines.append("## Key Takeaways")
            lines.extend(f"- {item}" for item in analysis["key_takeaways"])
            lines.append("")

        if analysis.get("common_concepts"):
            lines.append("## Common Concepts Across Documents")
            for concept in analysis["common_concepts"]:
                docs = ", ".join(concept.get("documents", []))
                lines.append(
                    f"- **{concept.get('concept', '')}** [{concept.get('importance', 'Medium')}] "
                    f"({docs}): {concept.get('explanation', '')}"
                )
            lines.append("")

        if analysis.get("document_analyses"):
            lines.append("## Per-Document Breakdown")
            for doc in analysis["document_analyses"]:
                lines.append(f"### {doc.get('filename', 'Document')}")
                lines.append(f"**Core Thesis:** {doc.get('core_thesis', '')}")
                lines.append(f"**Methodology:** {doc.get('methodology', '')}")
                if doc.get("key_findings"):
                    lines.append("**Key Findings:**")
                    lines.extend(f"- {f}" for f in doc["key_findings"])
                if doc.get("limitations"):
                    lines.append("**Limitations:**")
                    lines.extend(f"- {lim}" for lim in doc["limitations"])
                lines.append("")

    if gaps:
        lines.extend(["## Research Gap Detection", gaps.get("executive_summary", ""), ""])
        for gap in gaps.get("methodological_gaps", []):
            lines.append(
                f"- **[{gap.get('severity', 'Medium')}] {gap.get('title', '')}**: "
                f"{gap.get('description', '')} *Next step:* {gap.get('suggested_direction', '')}"
            )
        lines.append("")
        if gaps.get("open_research_questions"):
            lines.append("## Open Research Questions")
            for q in gaps["open_research_questions"]:
                lines.append(
                    f"- **[{q.get('priority', 'Medium')}] {q.get('question', '')}** — "
                    f"{q.get('rationale', '')}"
                )

    return "\n".join(lines).strip()
