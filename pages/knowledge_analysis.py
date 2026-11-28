"""Knowledge Analysis page: Structured breakdown, common concepts, and research gaps."""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from modules.ai_analysis import (
    AIServiceError,
    analyze_documents,
    format_analysis_markdown,
    is_api_configured,
)
from modules.history_db import HistoryStorageError, save_research_session
from modules.research_gap import build_coverage_heatmap, detect_research_gaps


def _badge_class(level: str) -> str:
    norm = (level or "").strip().lower()
    if norm == "high":
        return "kx-badge-high"
    if norm == "low":
        return "kx-badge-low"
    return "kx-badge-medium"


def _build_concept_frequency_chart(common_concepts: list[dict]) -> go.Figure | None:
    if not common_concepts:
        return None
    names = [c.get("concept", "Concept") for c in common_concepts]
    counts = [len(c.get("documents", [])) for c in common_concepts]
    colors = [
        "#f87171" if (c.get("importance") == "High")
        else "#38bdf8" if (c.get("importance") == "Medium")
        else "#34d399"
        for c in common_concepts
    ]

    fig = go.Figure(
        data=go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{n} doc(s)" for n in counts],
            textposition="auto",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        height=max(260, 44 * len(names) + 80),
        margin=dict(l=16, r=20, t=36, b=16),
        title=dict(text="Common Concept Coverage by Document Count", font=dict(size=15, color="#f1f5f9")),
        xaxis=dict(title="Documents Referencing Concept", dtick=1, gridcolor="#1e293b"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


st.markdown(
    """
    <div class="kx-hero">
        <h1>Knowledge Analysis &amp; Gap Detection</h1>
        <p>Structured paper breakdowns, shared cross-document concepts, and AI-detected research gaps.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

docs = st.session_state.get("documents", [])
analysis_existing = st.session_state.get("analysis_result")
gaps_existing = st.session_state.get("gap_report")
history_meta = st.session_state.get("active_history_session")

if history_meta:
    hb1, hb2 = st.columns([4.5, 1.2])
    with hb1:
        st.info(
            f"Viewing saved Research History session: **{history_meta.get('research_topic', '')}** "
            f"(`{history_meta.get('session_id', '')}` · {history_meta.get('created_at', '')})"
        )
    with hb2:
        if st.button("Close History View", width="stretch"):
            st.session_state.active_history_session = None
            st.session_state.current_session_id = None
            st.session_state.analysis_result = None
            st.session_state.gap_report = None
            st.session_state.knowledge_graph_data = None
            st.rerun()

if not docs and not (analysis_existing or gaps_existing):
    st.warning("No PDFs loaded. Upload one or more research PDFs in the Dashboard or sidebar first.")
    st.stop()

configured, _ = is_api_configured(st.session_state.get("api_key_override"))
if not configured and docs:
    st.info("Configure `GEMINI_API_KEY` in `.env` (or enter a session key in the sidebar) to run AI analysis.")

btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1.5])
with btn_col1:
    run_analysis_btn = st.button(
        "Generate Structured Analysis",
        type="primary",
        width="stretch",
        disabled=not (configured and docs),
    )
with btn_col2:
    run_gaps_btn = st.button(
        "Detect Research Gaps",
        width="stretch",
        disabled=not (configured and docs),
    )
with btn_col3:
    analysis_data = st.session_state.get("analysis_result")
    gap_data = st.session_state.get("gap_report")
    if analysis_data or gap_data:
        md_report = format_analysis_markdown(analysis_data or {}, gap_data)
        st.download_button(
            "Export Report (.md)",
            data=md_report,
            file_name="knowledge_x_report.md",
            mime="text/markdown",
            width="stretch",
        )

if run_analysis_btn:
    with st.spinner("Synthesizing structured analysis and common concepts..."):
        try:
            result = analyze_documents(
                docs,
                model=st.session_state.selected_model,
                api_key=st.session_state.api_key_override,
                allow_fallback=st.session_state.get("auto_fallback", True),
            )
            st.session_state.analysis_result = result
            st.session_state.active_history_session = None
            saved_id = save_research_session(
                documents=docs,
                analysis_results=result,
                research_gaps=st.session_state.get("gap_report"),
                knowledge_graph=st.session_state.get("knowledge_graph_data"),
                session_id=st.session_state.get("current_session_id"),
                user_id=(st.session_state.get("auth_user") or {}).get("user_id"),
            )
            st.session_state.current_session_id = saved_id
            if result.get("_fallback_note"):
                st.info(result["_fallback_note"])
            st.success(f"Structured knowledge analysis generated and saved to Research History (`{saved_id}`).")
        except (AIServiceError, HistoryStorageError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected analysis error: {exc}")

if run_gaps_btn:
    with st.spinner("Scanning documents for methodological gaps and open questions..."):
        try:
            gap_res = detect_research_gaps(
                docs,
                analysis_result=st.session_state.get("analysis_result"),
                model=st.session_state.selected_model,
                api_key=st.session_state.api_key_override,
                allow_fallback=st.session_state.get("auto_fallback", True),
            )
            st.session_state.gap_report = gap_res
            st.session_state.active_history_session = None
            saved_id = save_research_session(
                documents=docs,
                analysis_results=st.session_state.get("analysis_result"),
                research_gaps=gap_res,
                knowledge_graph=st.session_state.get("knowledge_graph_data"),
                session_id=st.session_state.get("current_session_id"),
                user_id=(st.session_state.get("auth_user") or {}).get("user_id"),
            )
            st.session_state.current_session_id = saved_id
            if gap_res.get("_fallback_note"):
                st.info(gap_res["_fallback_note"])
            st.success(f"Research gap detection completed and saved to Research History (`{saved_id}`).")
        except (AIServiceError, HistoryStorageError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected gap-detection error: {exc}")

analysis = st.session_state.get("analysis_result")
gaps = st.session_state.get("gap_report")
multi_llm = st.session_state.get("multi_llm_result")

if multi_llm and multi_llm.get("master_report"):
    master = multi_llm["master_report"]
    w3 = multi_llm.get("worker3_result") or {}
    usage = multi_llm.get("token_usage") or {}
    st.divider()
    st.subheader("KnowledgeX 2.0 — Multi-LLM Synthesis Report")
    st.caption(
        f"Status: **{multi_llm.get('status', 'completed').title()}** · "
        f"Cache Hits: **{multi_llm.get('cache_hits', 0)}** · "
        f"Tokens: **{usage.get('total_tokens', 0):,}**"
    )
    if multi_llm.get("failed_documents"):
        st.warning(
            f"Partial completion: {len(multi_llm['failed_documents'])} worker task(s) failed. "
            "Re-run Multi-LLM Synthesis in Dashboard to retry only failed documents."
        )

    m_tab_exec, m_tab_digests, m_tab_intel, m_tab_refs = st.tabs(
        ["Master Synthesis", "Document & Section Digests", "Comparison & Gaps", "Source Citations"]
    )
    with m_tab_exec:
        st.markdown(
            f"<div class='kx-card'><div class='kx-card-title'>Executive Summary</div>"
            f"<div class='kx-card-body'>{escape(master.get('executive_summary', ''))}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Synthesized Key Findings")
        for idx, item in enumerate(master.get("synthesized_findings", []), start=1):
            cits = ", ".join(
                f"{c.get('filename', '')} (p. {c.get('page_number', '?')})"
                for c in item.get("citations", [])
            )
            cit_html = f"<div class='kx-badge-row'><span class='kx-badge kx-badge-info'>{escape(cits)}</span></div>" if cits else ""
            st.markdown(
                f"<div class='kx-card'><div class='kx-card-body'><b>{idx}.</b> {escape(item.get('claim', ''))}</div>{cit_html}</div>",
                unsafe_allow_html=True,
            )

    with m_tab_digests:
        for dig in master.get("document_digests", []):
            pages_str = ", ".join(str(p) for p in dig.get("key_pages", []))
            pages_badge = f"<div class='kx-badge-row'><span class='kx-badge'>Key pages: {escape(pages_str)}</span></div>" if pages_str else ""
            st.markdown(
                f"<div class='kx-card'><div class='kx-card-title'>{escape(dig.get('filename', ''))}</div>"
                f"<div class='kx-card-body'>{escape(dig.get('summary', ''))}</div>{pages_badge}</div>",
                unsafe_allow_html=True,
            )
        for w2 in multi_llm.get("worker2_results", []):
            with st.expander(f"Section-by-Section Summary: {w2.get('filename', '')}", expanded=False):
                for sec in w2.get("section_summaries", []):
                    p_str = ", ".join(str(p) for p in sec.get("page_numbers", []))
                    st.markdown(f"**{sec.get('section_title', 'Section')}** *(pp. {p_str})* — {sec.get('summary', '')}")

    with m_tab_intel:
        if w3.get("cross_document_insights"):
            st.info(w3["cross_document_insights"])
        st.markdown("#### Common Concepts")
        for c in master.get("common_concepts", []):
            docs_str = ", ".join(c.get("documents", []))
            st.markdown(f"- **{c.get('concept', '')}** [{c.get('importance', 'Medium')}] ({docs_str}): {c.get('explanation', '')}")
        if w3.get("similarities"):
            st.markdown("#### Similarities")
            for sim in w3["similarities"]:
                st.markdown(f"- {sim}")
        if w3.get("contradictions"):
            st.markdown("#### Contradictions")
            for con in w3["contradictions"]:
                st.markdown(f"- {con}")
        if master.get("contradictions_and_gaps"):
            st.markdown("#### Contradictions & Research Gaps")
            for gap in master["contradictions_and_gaps"]:
                st.markdown(f"- {gap}")
        if master.get("open_research_questions"):
            st.markdown("#### Open Research Questions")
            for q in master["open_research_questions"]:
                st.markdown(f"- {q}")

    with m_tab_refs:
        refs = master.get("source_references", [])
        if not refs:
            st.info("No explicit source references were attached.")
        else:
            for ref in refs:
                st.markdown(f"- **`{ref.get('filename', '')}`** — Page {ref.get('page_number', '?')}")

if not analysis and not gaps:
    if not (multi_llm and multi_llm.get("master_report")):
        st.markdown(
            f"""
            <div class="kx-card">
                <div class="kx-card-title">Ready to Analyze {len(docs)} Loaded PDF(s)</div>
                <div class="kx-card-body">
                    Click <b>Generate Structured Analysis</b> to extract executive takeaways, per-paper breakdowns,
                    and common concepts, or click <b>Detect Research Gaps</b> to uncover missing baselines,
                    contradictions, and unexplored research directions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

tab_overview, tab_concepts, tab_papers, tab_gaps = st.tabs(
    ["Executive Synthesis", "Common Concepts", "Document Breakdowns", "Research Gaps"]
)

with tab_overview:
    if not analysis:
        st.info("Run **Generate Structured Analysis** to populate the Executive Synthesis.")
    else:
        st.markdown(
            f"""
            <div class="kx-card">
                <div class="kx-card-title">Corpus Overview</div>
                <div class="kx-card-body">{escape(analysis.get("corpus_overview", ""))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Key Takeaways")
        for idx, item in enumerate(analysis.get("key_takeaways", []), start=1):
            st.markdown(
                f"""
                <div class="kx-card">
                    <div class="kx-card-body"><b>{idx}.</b> {escape(item)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        glossary = analysis.get("terminology_glossary", [])
        if glossary:
            st.subheader("Core Terminology Glossary")
            for entry in glossary:
                st.markdown(
                    f"""
                    <div class="kx-card">
                        <div class="kx-card-title">{escape(entry.get("term", ""))}</div>
                        <div class="kx-card-body">{escape(entry.get("definition", ""))}</div>
                        <div class="kx-badge-row">
                            <span class="kx-badge kx-badge-info">{escape(entry.get("source_document", ""))}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_concepts:
    if not analysis:
        st.info("Run **Generate Structured Analysis** to identify common concepts across documents.")
    else:
        concepts = analysis.get("common_concepts", [])
        if not concepts:
            st.info("No common concepts were returned for this corpus.")
        else:
            fig = _build_concept_frequency_chart(concepts)
            if fig is not None:
                st.plotly_chart(fig, width="stretch", theme=None)

            for item in concepts:
                badge_cls = _badge_class(item.get("importance", "Medium"))
                doc_badges = "".join(
                    f"<span class='kx-badge kx-badge-info'>{escape(d)}</span>"
                    for d in item.get("documents", [])
                )
                st.markdown(
                    f"""
                    <div class="kx-card">
                        <div class="kx-card-title">
                            {escape(item.get("concept", ""))}
                            &nbsp;<span class="kx-badge {badge_cls}">{escape(item.get("importance", "Medium"))} Importance</span>
                        </div>
                        <div class="kx-card-body">{escape(item.get("explanation", ""))}</div>
                        <div class="kx-badge-row">{doc_badges}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_papers:
    if not analysis:
        st.info("Run **Generate Structured Analysis** to view per-document structured breakdowns.")
    else:
        for paper in analysis.get("document_analyses", []):
            with st.expander(paper.get("filename", "Document"), expanded=True):
                st.markdown(f"**Core Thesis:** {paper.get('core_thesis', '')}")
                st.markdown(f"**Methodology:** {paper.get('methodology', '')}")

                c_find, c_lim = st.columns(2)
                with c_find:
                    st.markdown("**Key Findings**")
                    for finding in paper.get("key_findings", []):
                        st.markdown(f"- {finding}")
                with c_lim:
                    st.markdown("**Limitations**")
                    for lim in paper.get("limitations", []):
                        st.markdown(f"- {lim}")

                terms = paper.get("key_terms", [])
                if terms:
                    badges = "".join(f"<span class='kx-badge'>{escape(t)}</span>" for t in terms)
                    st.markdown(f"<div class='kx-badge-row'>{badges}</div>", unsafe_allow_html=True)

with tab_gaps:
    if not gaps:
        st.info("Click **Detect Research Gaps** above to identify blind spots, tensions, and open questions.")
    else:
        st.markdown(
            f"""
            <div class="kx-card">
                <div class="kx-card-title">Gap Detection Summary</div>
                <div class="kx-card-body">{escape(gaps.get("executive_summary", ""))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        heatmap = build_coverage_heatmap(gaps, docs)
        if heatmap is not None:
            st.plotly_chart(heatmap, width="stretch", theme=None)

        st.subheader("Methodological & Evidence Gaps")
        for gap in gaps.get("methodological_gaps", []):
            sev_cls = _badge_class(gap.get("severity", "Medium"))
            affected = "".join(
                f"<span class='kx-badge kx-badge-info'>{escape(d)}</span>"
                for d in gap.get("affected_documents", [])
            )
            st.markdown(
                f"""
                <div class="kx-card">
                    <div class="kx-card-title">
                        {escape(gap.get("title", ""))}
                        &nbsp;<span class="kx-badge {sev_cls}">{escape(gap.get("severity", "Medium"))} Severity</span>
                    </div>
                    <div class="kx-card-body">
                        {escape(gap.get("description", ""))}<br><br>
                        <b>Observed Evidence:</b> {escape(gap.get("evidence", ""))}<br>
                        <b>Suggested Direction:</b> {escape(gap.get("suggested_direction", ""))}
                    </div>
                    <div class="kx-badge-row">{affected}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        tensions = gaps.get("contradictions_and_tensions", [])
        if tensions:
            st.subheader("Contradictions & Trade-offs")
            for tension in tensions:
                involved = "".join(
                    f"<span class='kx-badge kx-badge-info'>{escape(d)}</span>"
                    for d in tension.get("documents_involved", [])
                )
                st.markdown(
                    f"""
                    <div class="kx-card">
                        <div class="kx-card-title">{escape(tension.get("topic", ""))}</div>
                        <div class="kx-card-body">{escape(tension.get("conflicting_perspectives", ""))}</div>
                        <div class="kx-badge-row">{involved}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        open_qs = gaps.get("open_research_questions", [])
        if open_qs:
            st.subheader("Open Research Questions")
            for q in open_qs:
                pri_cls = _badge_class(q.get("priority", "Medium"))
                st.markdown(
                    f"""
                    <div class="kx-card">
                        <div class="kx-card-title">
                            {escape(q.get("question", ""))}
                            &nbsp;<span class="kx-badge {pri_cls}">{escape(q.get("priority", "Medium"))} Priority</span>
                        </div>
                        <div class="kx-card-body">{escape(q.get("rationale", ""))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
