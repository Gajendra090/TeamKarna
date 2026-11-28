"""AI Ambiguity Detector: Question classification, interactive clarification, rewriting, and grounded answering."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from modules.ai_analysis import (
    AIServiceError,
    DEFAULT_MODEL,
    answer_research_question,
    generate_structured_json,
)
from modules.pdf_reader import build_corpus_context


class AmbiguityClassification(BaseModel):
    classification: Literal["Clear", "Ambiguous", "Insufficient Context"]
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="Confidence in the ambiguity classification."
    )
    unclear_phrases: list[str] = Field(
        description="Exact unclear words or phrases from the question when Ambiguous or missing referent; empty otherwise."
    )
    ambiguity_reason: str = Field(
        description="Non-empty explanation of why the question is ambiguous or lacks context; empty string when Clear."
    )
    interpretations: list[str] = Field(
        description="2 to 4 concrete meanings or interpretations when Ambiguous or when a broad/polysemous query has multiple possible meanings; empty list when Clear."
    )
    missing_information: list[str] = Field(
        description="Specific missing constraints, document/author names, or context when Insufficient Context; empty list otherwise."
    )
    clarifying_prompt: str = Field(
        description="Concise question asking the user to select an interpretation or supply missing info; empty string when Clear."
    )


class ResolvedQuestionResult(BaseModel):
    resolved_question: str = Field(
        description="Clear, self-contained natural-language question incorporating the user's clarification."
    )
    resolution_summary: str = Field(
        description="Brief one-sentence explanation of how the ambiguity or missing context was resolved."
    )


def _format_document_inventory(documents: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {d['filename']} (Title: {d.get('title', d['filename'])}, "
        f"Author: {d.get('author', 'Unknown')}, Pages: {d.get('page_count', 0)})"
        for d in documents
    )


def _format_history_snippet(
    chat_history: list[dict[str, Any]] | None = None,
    clarification_history: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    for idx, msg in enumerate((chat_history or [])[-6:], start=1):
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("resolved_question") or msg.get("content") or "").strip()
        if content:
            lines.append(f"[Turn {idx}] {role}: {content[:500]}")
        ans = str(msg.get("answer") or "").strip()
        if ans:
            lines.append(f"[Turn {idx}] ASSISTANT_SUMMARY: {ans[:300]}")
    for clar in (clarification_history or [])[-3:]:
        lines.append(
            f"PRIOR_CLARIFICATION_ASKED: {clar.get('clarifying_prompt', '')} | "
            f"USER_CLARIFIED: {clar.get('user_clarification', '')}"
        )
    return "\n".join(lines) if lines else "None"


def analyze_question_ambiguity(
    question: str,
    documents: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
    clarification_history: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Classify a question as Clear, Ambiguous, or Insufficient Context."""
    clean_question = (question or "").strip()
    if not clean_question:
        raise AIServiceError("Please enter a question.")
    if not documents:
        raise AIServiceError("Upload at least one PDF before using the Ambiguity Detector.")

    doc_inventory = _format_document_inventory(documents)
    corpus_preview = build_corpus_context(documents, max_chars=24000)
    history_block = _format_history_snippet(chat_history, clarification_history)

    system_instruction = (
        "You are the KNOWLEDGE X AI Ambiguity Detector. Analyze a user's question against "
        "the loaded research PDFs, document count, and recent conversation turns BEFORE generating an answer.\n\n"
        "Classify into EXACTLY one of:\n"
        "1. 'Clear':\n"
        "   - Specific intent, unambiguous entities/documents/concepts, and sufficient context.\n"
        "   - ALSO 'Clear' when a pronoun or follow-up ('its', 'it', 'they', 'that method') "
        "unambiguously refers to a SINGLE primary subject explicitly named in the most recent "
        "[Turn] of CONVERSATION & PRIOR CLARIFICATIONS (do NOT treat unrelated topics inside "
        "DOCUMENT EXCERPTS as competing conversational referents).\n"
        "   - Broad but well-defined questions (e.g., summarizing a named PDF or explaining advantages "
        "of a named concept in a named domain like 'normalization in DBMS') are Clear.\n"
        "   - Never request unnecessary clarification for already-clear questions.\n\n"
        "2. 'Ambiguous':\n"
        "   - Vague modifiers ('better', 'best', 'faster', 'important') or unreferenced pronouns "
        "when multiple competing subjects exist in the same turn/corpus.\n"
        "   - Broad, polysemous, or multi-interpretation questions lacking specific scope — even if the "
        "term is absent from the loaded PDFs "
        "(e.g., 'What is Java?' -> Java programming language/JVM vs Java island vs Java coffee; "
        "'What is the impact of technology?' -> economic, social, environmental, healthcare, or specific technologies; "
        "'Compare these two documents.' when MORE than 2 documents are loaded and no pair is specified).\n"
        "   - MUST provide exact unclear_phrases from the question, a clear non-empty ambiguity_reason "
        "(mentioning if the term is absent from the loaded PDFs), and 2 to 4 concrete, meaningful interpretations.\n\n"
        "3. 'Insufficient Context':\n"
        "   - Missing mandatory context, target document/author, constraints, or missing referent with NO prior conversation:\n"
        "     * Pronouns with zero prior conversation turns (e.g., 'What are its advantages?' when CONVERSATION is None).\n"
        "     * Asking to compare two documents when LOADED DOCUMENTS count < 2.\n"
        "     * Asking 'What did the author conclude?' when multiple documents exist or author/document is unspecified.\n"
        "     * Asking about topics completely absent from the loaded PDFs (e.g., 'Explain Python.' when only DBMS PDFs are loaded) "
        "or asking constraint-dependent advice without constraints ('Will this meet my latency budget?').\n"
        "   - MUST provide a non-empty ambiguity_reason explaining what context is missing, exact missing_information items, "
        "and a helpful clarifying_prompt instead of guessing. If the query also mentions a broad or polysemous term with multiple "
        "meanings (e.g., 'What is Java?' -> programming language, island, coffee), ALSO list 2 to 4 meanings in interpretations.\n\n"
        "Never repeat a clarification already answered in PRIOR_CLARIFICATION_ASKED."
    )

    prompt = (
        f"LOADED DOCUMENTS COUNT: {len(documents)}\n"
        f"LOADED DOCUMENTS:\n{doc_inventory}\n\n"
        f"CONVERSATION & PRIOR CLARIFICATIONS:\n{history_block}\n\n"
        f"DOCUMENT EXCERPTS:\n{corpus_preview}\n\n"
        f"USER QUESTION TO ANALYZE:\n{clean_question}"
    )

    result = generate_structured_json(
        prompt=prompt,
        schema=AmbiguityClassification,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.1,
        allow_fallback=allow_fallback,
    )

    # Deterministic check: asking to compare two documents when <2 PDFs are loaded cannot be Clear
    if len(documents) < 2 and re.search(
        r"\b(compare|difference|between)\b.*\b(two|both)\s+(documents?|papers?|pdfs?|files?)\b",
        clean_question,
        re.IGNORECASE,
    ):
        result["classification"] = "Insufficient Context"
        result["unclear_phrases"] = result.get("unclear_phrases") or ["two documents"]
        result["ambiguity_reason"] = (
            f"You asked to compare two documents, but only {len(documents)} PDF "
            f"('{documents[0]['filename']}') is currently loaded."
        )
        result["missing_information"] = [
            "A second PDF document uploaded to the workspace",
            "Specific comparison criteria (once two documents are available)",
        ]
        result["clarifying_prompt"] = (
            "Please upload a second document first, or tell me which topics within the loaded PDF you want compared."
        )

    label = result.get("classification", "Clear")
    if result.get("confidence") not in ("High", "Medium", "Low"):
        result["confidence"] = "Medium"

    if label == "Clear":
        result["unclear_phrases"] = []
        result["ambiguity_reason"] = ""
        result["interpretations"] = []
        result["missing_information"] = []
        result["clarifying_prompt"] = ""
    elif label == "Ambiguous":
        interps = [str(x).strip() for x in (result.get("interpretations") or []) if str(x).strip()]
        if len(interps) < 2:
            doc_names = [d["filename"] for d in documents]
            if len(doc_names) >= 2:
                interps = [
                    f"Focus on {doc_names[0]}",
                    f"Focus on {doc_names[1]}",
                    "Compare across all uploaded documents",
                ]
            else:
                interps = [
                    "Provide a conceptual definition and mechanism",
                    "Provide practical trade-offs, limitations, and metrics",
                ]
        result["interpretations"] = interps[:4]
        result["unclear_phrases"] = [
            str(x).strip() for x in (result.get("unclear_phrases") or []) if str(x).strip()
        ]
        if not (result.get("ambiguity_reason") or "").strip():
            result["ambiguity_reason"] = (
                "The question has multiple plausible interpretations without additional scope."
            )
        if not (result.get("clarifying_prompt") or "").strip():
            result["clarifying_prompt"] = "Please select one of the interpretations above or clarify what you mean."
        result["missing_information"] = []
    else:
        # Insufficient Context
        interps = [str(x).strip() for x in (result.get("interpretations") or []) if str(x).strip()]
        result["interpretations"] = interps[:4] if len(interps) >= 2 else []
        result["unclear_phrases"] = [
            str(x).strip() for x in (result.get("unclear_phrases") or []) if str(x).strip()
        ]
        missing = [
            str(x).strip() for x in (result.get("missing_information") or []) if str(x).strip()
        ]
        if not missing:
            missing = ["Additional context, target document, or specific constraints"]
        result["missing_information"] = missing
        if not (result.get("clarifying_prompt") or "").strip():
            result["clarifying_prompt"] = "Please provide the missing context or constraints so I can answer accurately."
        if not (result.get("ambiguity_reason") or "").strip():
            result["ambiguity_reason"] = (
                f"Additional context is needed before answering ({'; '.join(missing[:2])})."
            )

    return result


def rewrite_clarified_question(
    original_question: str,
    user_clarification: str,
    classification_info: dict[str, Any],
    documents: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Rewrite an ambiguous or underspecified question into a clear, self-contained question."""
    clean_orig = (original_question or "").strip()
    clean_clar = (user_clarification or "").strip()
    if not clean_orig:
        raise AIServiceError("Original question cannot be empty.")
    if not clean_clar:
        raise AIServiceError("Please select an interpretation or enter your clarification.")

    doc_inventory = _format_document_inventory(documents)
    history_block = _format_history_snippet(chat_history)

    system_instruction = (
        "You rewrite ambiguous or incomplete research questions into a single, clear, "
        "self-contained question using the user's clarification and loaded document context. "
        "Preserve the user's exact intent without inventing unsupported assumptions."
    )

    context_meta = {
        "classification": classification_info.get("classification"),
        "unclear_phrases": classification_info.get("unclear_phrases", []),
        "ambiguity_reason": classification_info.get("ambiguity_reason", ""),
        "missing_information": classification_info.get("missing_information", []),
    }

    prompt = (
        f"LOADED DOCUMENTS:\n{doc_inventory}\n\n"
        f"RECENT CONVERSATION:\n{history_block}\n\n"
        f"DETECTED ISSUE:\n{json.dumps(context_meta, indent=2)}\n\n"
        f"ORIGINAL QUESTION:\n{clean_orig}\n\n"
        f"USER CLARIFICATION:\n{clean_clar}\n\n"
        "Rewrite this into a crisp, unambiguous research question."
    )

    return generate_structured_json(
        prompt=prompt,
        schema=ResolvedQuestionResult,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.1,
        allow_fallback=allow_fallback,
    )


def process_initial_question(
    question: str,
    documents: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
    clarification_history: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Analyze question ambiguity; answer immediately if Clear, otherwise return clarification request."""
    analysis = analyze_question_ambiguity(
        question=question,
        documents=documents,
        chat_history=chat_history,
        clarification_history=clarification_history,
        model=model,
        api_key=api_key,
        allow_fallback=allow_fallback,
    )

    if analysis["classification"] == "Clear":
        qa_result = answer_research_question(
            question=question,
            documents=documents,
            chat_history=chat_history,
            model=model,
            api_key=api_key,
            allow_fallback=allow_fallback,
        )
        return {
            "status": "answered",
            "original_question": question.strip(),
            "resolved_question": question.strip(),
            "classification_info": analysis,
            "user_clarification": "",
            "resolution_summary": "Question was clear; answered directly from uploaded PDFs.",
            "answer": qa_result.get("answer", ""),
            "citations": qa_result.get("citations", []),
            "confidence": qa_result.get("confidence", "Medium"),
            "follow_up_questions": qa_result.get("follow_up_questions", []),
            "model_used": qa_result.get("_model_used", analysis.get("_model_used", model)),
            "fallback_note": qa_result.get("_fallback_note") or analysis.get("_fallback_note", ""),
        }

    return {
        "status": "needs_clarification",
        "original_question": question.strip(),
        "classification_info": analysis,
        "model_used": analysis.get("_model_used", model),
        "fallback_note": analysis.get("_fallback_note", ""),
    }


def complete_clarified_question(
    original_question: str,
    user_clarification: str,
    classification_info: dict[str, Any],
    documents: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Rewrite the clarified question and generate the grounded final answer from PDFs."""
    rewritten = rewrite_clarified_question(
        original_question=original_question,
        user_clarification=user_clarification,
        classification_info=classification_info,
        documents=documents,
        chat_history=chat_history,
        model=model,
        api_key=api_key,
        allow_fallback=allow_fallback,
    )
    resolved_q = rewritten.get("resolved_question", "").strip() or (
        f"{original_question.strip()} ({user_clarification.strip()})"
    )

    qa_result = answer_research_question(
        question=resolved_q,
        documents=documents,
        chat_history=chat_history,
        model=model,
        api_key=api_key,
        allow_fallback=allow_fallback,
    )

    return {
        "status": "answered",
        "original_question": original_question.strip(),
        "resolved_question": resolved_q,
        "classification_info": classification_info,
        "user_clarification": user_clarification.strip(),
        "resolution_summary": rewritten.get("resolution_summary", ""),
        "answer": qa_result.get("answer", ""),
        "citations": qa_result.get("citations", []),
        "confidence": qa_result.get("confidence", "Medium"),
        "follow_up_questions": qa_result.get("follow_up_questions", []),
        "model_used": qa_result.get("_model_used", rewritten.get("_model_used", model)),
        "fallback_note": qa_result.get("_fallback_note") or rewritten.get("_fallback_note", ""),
    }
