"""Knowledge Map page: Interactive NetworkX + Plotly visual knowledge graph."""

from __future__ import annotations

from html import escape

import streamlit as st

from modules.ai_analysis import AIServiceError, is_api_configured
from modules.history_db import HistoryStorageError, save_research_session
from modules.knowledge_map import (
    NODE_COLORS,
    build_networkx_graph,
    build_structural_graph_data,
    extract_ai_knowledge_graph,
    render_plotly_knowledge_map,
    summarize_graph_metrics,
)

st.markdown(
    """
    <div class="kx-hero">
        <h1>Visual Knowledge Map</h1>
        <p>Explore concepts, methods, documents, and research gaps as an interactive NetworkX &amp; Plotly graph.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

docs = st.session_state.get("documents", [])
existing_graph = st.session_state.get("knowledge_graph_data")
history_meta = st.session_state.get("active_history_session")

if history_meta:
    st.info(
        f"Viewing Knowledge Map for saved session: **{history_meta.get('research_topic', '')}** "
        f"(`{history_meta.get('session_id', '')}`)"
    )

if not docs and not existing_graph:
    st.warning("No PDFs loaded. Upload one or more research PDFs in the Dashboard or sidebar first.")
    st.stop()

configured, _ = is_api_configured(st.session_state.get("api_key_override"))

c_ai, c_local = st.columns(2)
with c_ai:
    gen_ai_btn = st.button(
        "Generate AI Semantic Knowledge Map",
        type="primary",
        width="stretch",
        disabled=not (configured and docs),
        help="Uses Gemini to extract semantic entities and typed cross-document edges.",
    )
with c_local:
    gen_local_btn = st.button(
        "Build Structural / Keyword Map",
        width="stretch",
        disabled=not (docs or st.session_state.get("analysis_result")),
        help="Builds a local NetworkX graph from existing analysis or PyMuPDF document keywords.",
    )

if gen_ai_btn:
    with st.spinner("Extracting semantic knowledge graph with Gemini..."):
        try:
            kg_res = extract_ai_knowledge_graph(
                documents=docs,
                analysis_result=st.session_state.get("analysis_result"),
                gap_report=st.session_state.get("gap_report"),
                model=st.session_state.selected_model,
                api_key=st.session_state.api_key_override,
                allow_fallback=st.session_state.get("auto_fallback", True),
            )
            st.session_state.knowledge_graph_data = kg_res
            if st.session_state.get("current_session_id"):
                try:
                    save_research_session(
                        documents=docs,
                        analysis_results=st.session_state.get("analysis_result"),
                        research_gaps=st.session_state.get("gap_report"),
                        knowledge_graph=kg_res,
                        session_id=st.session_state.get("current_session_id"),
                        user_id=(st.session_state.get("auth_user") or {}).get("user_id"),
                    )
                except HistoryStorageError:
                    pass
            if kg_res.get("_fallback_note"):
                st.info(kg_res["_fallback_note"])
            st.success("AI semantic knowledge map generated.")
        except AIServiceError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected knowledge map error: {exc}")

if gen_local_btn or st.session_state.get("knowledge_graph_data") is None:
    if docs:
        st.session_state.knowledge_graph_data = build_structural_graph_data(
            documents=docs,
            analysis_result=st.session_state.get("analysis_result"),
            gap_report=st.session_state.get("gap_report"),
        )
    elif st.session_state.get("analysis_result"):
        saved_analysis = st.session_state["analysis_result"]
        pseudo_docs = [
            {"filename": d.get("filename", "Document"), "title": d.get("filename", "Document"), "page_count": 0, "word_count": 0, "full_text": ""}
            for d in saved_analysis.get("document_analyses", [])
            if d.get("filename")
        ]
        st.session_state.knowledge_graph_data = build_structural_graph_data(
            documents=pseudo_docs,
            analysis_result=saved_analysis,
            gap_report=st.session_state.get("gap_report"),
        )

graph_data = st.session_state.get("knowledge_graph_data") or {"nodes": [], "edges": []}
graph = build_networkx_graph(graph_data)
metrics = summarize_graph_metrics(graph)

st.caption(f"Graph mode: **{graph_data.get('source_mode', 'Structural Graph')}**")

f_col1, f_col2 = st.columns([2.5, 1.5])
with f_col1:
    visible_types = st.multiselect(
        "Visible node types",
        options=list(NODE_COLORS.keys()),
        default=list(NODE_COLORS.keys()),
    )
with f_col2:
    spacing = st.slider("Node spacing", min_value=0.8, max_value=2.8, value=1.5, step=0.1)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Nodes", metrics["nodes"])
with k2:
    st.metric("Connections", metrics["edges"])
with k3:
    st.metric("Clusters", metrics["components"])
with k4:
    st.metric("Graph Density", metrics["density"])

fig = render_plotly_knowledge_map(
    graph,
    visible_types=visible_types,
    layout_spacing=spacing,
)
st.plotly_chart(fig, width="stretch", theme=None)

col_bridge, col_edges = st.columns([1.2, 1.8])
with col_bridge:
    st.subheader("Top Bridge Concepts")
    top_concepts = metrics.get("top_concepts", [])
    if not top_concepts:
        st.caption("No concept nodes available.")
    else:
        for item in top_concepts:
            st.markdown(
                f"""
                <div class="kx-card">
                    <div class="kx-card-title">
                        {escape(item['label'])}
                        &nbsp;<span class="kx-badge kx-badge-info">{item['connections']} links</span>
                    </div>
                    <div class="kx-card-body">{escape(item.get('summary') or item['type'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_edges:
    st.subheader("Graph Connections")
    edges = graph_data.get("edges", [])
    if not edges:
        st.caption("No connections in current graph.")
    else:
        rows_html = "".join(
            f"<tr><td>{escape(e.get('source', ''))}</td>"
            f"<td><span class='kx-badge'>{escape(e.get('relationship', ''))}</span></td>"
            f"<td>{escape(e.get('target', ''))}</td></tr>"
            for e in edges[:40]
        )
        st.markdown(
            f"""
            <div class="kx-card" style="max-height: 360px; overflow-y: auto;">
                <table class="kx-table">
                    <thead><tr><th>Source</th><th>Relationship</th><th>Target</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
