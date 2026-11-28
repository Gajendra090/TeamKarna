"""Cross-document research gap detection and topic coverage visualization."""

from __future__ import annotations

import json
from typing import Any, Literal

import plotly.graph_objects as go
from pydantic import BaseModel, Field

from modules.ai_analysis import AIServiceError, DEFAULT_MODEL, generate_structured_json
from modules.pdf_reader import build_corpus_context


class GapItem(BaseModel):
    title: str
    description: str
    affected_documents: list[str]
    evidence: str = Field(description="Supporting evidence or missing element observed in the text.")
    severity: Literal["High", "Medium", "Low"]
    suggested_direction: str = Field(description="Concrete experiment, study, or resolution step.")


class TensionItem(BaseModel):
    topic: str
    conflicting_perspectives: str
    documents_involved: list[str]


class OpenQuestion(BaseModel):
    question: str
    rationale: str
    priority: Literal["High", "Medium", "Low"]


class TopicCoverage(BaseModel):
    topic: str = Field(description="Major research theme, metric, or subproblem.")
    covered_by: list[str] = Field(description="Exact filenames that address this topic.")
    missing_in: list[str] = Field(description="Exact filenames that omit or only marginally touch it.")


class ResearchGapReport(BaseModel):
    executive_summary: str
    methodological_gaps: list[GapItem]
    contradictions_and_tensions: list[TensionItem]
    open_research_questions: list[OpenQuestion]
    coverage_matrix: list[TopicCoverage]


def detect_research_gaps(
    documents: list[dict[str, Any]],
    analysis_result: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Identify methodological blind spots, tensions, and open research questions."""
    if not documents:
        raise AIServiceError("Upload at least one PDF before running research gap detection.")

    corpus_text = build_corpus_context(documents, max_chars=110000)
    filenames = [doc["filename"] for doc in documents]

    analysis_hint = ""
    if analysis_result:
        compact_hint = {
            "corpus_overview": analysis_result.get("corpus_overview", ""),
            "common_concepts": analysis_result.get("common_concepts", []),
            "limitations_by_doc": [
                {
                    "filename": d.get("filename"),
                    "limitations": d.get("limitations", []),
                }
                for d in analysis_result.get("document_analyses", [])
            ],
        }
        analysis_hint = (
            "\nPREVIOUS STRUCTURED ANALYSIS CONTEXT:\n"
            f"{json.dumps(compact_hint, indent=2)}\n"
        )

    system_instruction = (
        "You are a senior scientific reviewer identifying genuine research gaps, missing baselines, "
        "untested assumptions, cross-paper contradictions, and unexplored future directions. "
        "Stay strictly grounded in the uploaded papers."
    )

    prompt = (
        f"Inspect the following {len(documents)} research PDF(s): {', '.join(filenames)}.\n\n"
        "Identify:\n"
        "1. methodological_gaps: Concrete missing evaluations, scalability limits, data/domain gaps, "
        "or unverified assumptions.\n"
        "2. contradictions_and_tensions: Disagreements, trade-offs, or conflicting claims across papers "
        "(or internal trade-offs if only 1 paper is loaded).\n"
        "3. open_research_questions: High-value, actionable research questions emerging from these gaps.\n"
        "4. coverage_matrix: 5 to 8 key research topics/dimensions showing which exact filenames cover "
        "each topic vs which filenames do not.\n"
        f"{analysis_hint}\n"
        f"DOCUMENTS:\n{corpus_text}"
    )

    return generate_structured_json(
        prompt=prompt,
        schema=ResearchGapReport,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.25,
        allow_fallback=allow_fallback,
    )


def build_coverage_heatmap(
    gap_report: dict[str, Any],
    documents: list[dict[str, Any]],
) -> go.Figure | None:
    """Render a Plotly heatmap showing which documents cover vs miss key research topics."""
    matrix = gap_report.get("coverage_matrix") or []
    filenames = [doc["filename"] for doc in (documents or [])]
    if not filenames and matrix:
        inferred: list[str] = []
        seen: set[str] = set()
        for row in matrix:
            for name in list(row.get("covered_by", [])) + list(row.get("missing_in", [])):
                clean = str(name).strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    inferred.append(clean)
        filenames = inferred
    if not matrix or not filenames:
        return None

    topics = [row["topic"] for row in matrix]
    z_values: list[list[int]] = []
    hover_text: list[list[str]] = []

    for row in matrix:
        covered = {name.strip() for name in row.get("covered_by", [])}
        row_z: list[int] = []
        row_hover: list[str] = []
        for fname in filenames:
            is_covered = fname in covered
            row_z.append(1 if is_covered else 0)
            status = "Covered" if is_covered else "Gap / Not Addressed"
            row_hover.append(f"Topic: {row['topic']}<br>Document: {fname}<br>Status: {status}")
        z_values.append(row_z)
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=filenames,
            y=topics,
            text=hover_text,
            hoverinfo="text",
            colorscale=[[0.0, "#1e293b"], [1.0, "#38bdf8"]],
            showscale=False,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        margin=dict(l=16, r=16, t=36, b=16),
        height=max(280, 52 * len(topics) + 90),
        title=dict(text="Topic Coverage Matrix Across Documents", font=dict(size=15, color="#f1f5f9")),
        xaxis=dict(side="top", tickfont=dict(color="#cbd5e1")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#cbd5e1")),
    )
    return fig
