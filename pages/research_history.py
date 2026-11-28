"""Research History page: User-isolated SQLite browser for research analyses and ambiguity resolutions."""

from __future__ import annotations

from datetime import date, timedelta
from html import escape

import streamlit as st

from modules.ai_analysis import format_analysis_markdown
from modules.history_db import (
    HistoryStorageError,
    delete_ambiguity_record,
    delete_research_session,
    list_ambiguity_records,
    list_research_sessions,
)
from modules.knowledge_map import build_structural_graph_data


def _resolve_date_bounds(preset: str, custom_range: tuple | date | None) -> tuple[date | None, date | None]:
    today = date.today()
    if preset == "Today":
        return today, today
    if preset == "Last 7 days":
        return today - timedelta(days=7), today
    if preset == "Last 30 days":
        return today - timedelta(days=30), today
    if preset == "Custom range":
        if isinstance(custom_range, tuple) and len(custom_range) == 2:
            return custom_range[0], custom_range[1]
        if isinstance(custom_range, date):
            return custom_range, custom_range
    return None, None


st.markdown(
    """
    <div class="kx-hero">
        <h1>Research History</h1>
        <p>Browse, search, reopen, and manage your saved research analyses and ambiguity resolutions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

auth_user = st.session_state.get("auth_user")
if not auth_user or not auth_user.get("user_id"):
    st.markdown(
        """
        <div class="kx-card">
            <div class="kx-card-title">Sign in Required for Personal Research History</div>
            <div class="kx-card-body">
                Research history is private to each authenticated Google account.<br>
                Click the profile icon at the top of the left sidebar to sign in with Google and view, reopen, or manage your saved sessions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

active_user_id = auth_user["user_id"]

tab_analyses, tab_ambiguity = st.tabs(["Research Analyses", "Ambiguity Resolutions"])

with tab_analyses:
    f_col1, f_col2, f_col3 = st.columns([2.4, 1.3, 1.3])
    with f_col1:
        search_query = st.text_input(
            "Search history",
            placeholder="Filter by topic, document name, or concept...",
            key="hist_search",
        )
    with f_col2:
        date_preset = st.selectbox(
            "Date filter",
            options=["All dates", "Today", "Last 7 days", "Last 30 days", "Custom range"],
            key="hist_date_preset",
        )

    custom_dates = None
    if date_preset == "Custom range":
        with f_col3:
            today = date.today()
            custom_dates = st.date_input(
                "Date range",
                value=(today - timedelta(days=30), today),
                key="hist_custom_dates",
            )
    else:
        with f_col3:
            st.caption("Chronological order (newest first)")

    date_from, date_to = _resolve_date_bounds(date_preset, custom_dates)

    try:
        all_sessions = list_research_sessions(user_id=active_user_id)
        filtered_sessions = list_research_sessions(
            search_query=search_query,
            date_from=date_from,
            date_to=date_to,
            user_id=active_user_id,
        )
    except HistoryStorageError as exc:
        st.error(str(exc))
        st.stop()

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Saved Sessions", len(all_sessions))
    with k2:
        st.metric("Matching Sessions", len(filtered_sessions))
    with k3:
        st.metric("Documents in View", sum(s.get("document_count", 0) for s in filtered_sessions))

    st.write("")

    if not all_sessions:
        st.markdown(
            """
            <div class="kx-card">
                <div class="kx-card-title">No Saved Research Sessions Yet</div>
                <div class="kx-card-body">
                    Run <b>Run Full AI Synthesis Pipeline</b> on the Dashboard or <b>Generate Structured Analysis</b>
                    in Knowledge Analysis. Completed analyses are automatically saved to your account.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not filtered_sessions:
        st.info("No saved sessions match your current search or date filters.")
    else:
        for session in filtered_sessions:
            sid = session["session_id"]
            topic = session.get("research_topic") or "Untitled Research Session"
            created_at = session.get("created_at", "")
            doc_names = session.get("document_names", [])
            doc_count = session.get("document_count", len(doc_names))
            analysis = session.get("analysis_results") or {}
            concepts = session.get("common_concepts") or analysis.get("common_concepts") or []
            gaps = session.get("research_gaps") or {}
            graph = session.get("knowledge_graph") or {}

            doc_badges = "".join(
                f"<span class='kx-badge kx-badge-info'>{escape(name)}</span>"
                for name in doc_names[:8]
            )
            if len(doc_names) > 8:
                doc_badges += f"<span class='kx-badge'>+{len(doc_names) - 8} more</span>"

            status_badges = []
            if analysis:
                status_badges.append("<span class='kx-badge kx-badge-ok'>Analysis Saved</span>")
            if concepts:
                status_badges.append(
                    f"<span class='kx-badge kx-badge-info'>{len(concepts)} Common Concepts</span>"
                )
            if gaps:
                gap_cnt = len(gaps.get("methodological_gaps", []))
                status_badges.append(
                    f"<span class='kx-badge kx-badge-medium'>{gap_cnt} Research Gaps</span>"
                )

            st.markdown(
                f"""
                <div class="kx-card">
                    <div class="kx-card-title">{escape(topic)}</div>
                    <div class="kx-card-body">
                        <b>Session ID:</b> <code>{escape(sid)}</code> &nbsp;&middot;&nbsp;
                        <b>Saved:</b> {escape(created_at)} &nbsp;&middot;&nbsp;
                        <b>Documents:</b> {doc_count}
                    </div>
                    <div class="kx-badge-row">{doc_badges}</div>
                    <div class="kx-badge-row">{''.join(status_badges)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            act1, act2, act3 = st.columns([1.8, 1.5, 1.1])
            with act1:
                if st.button("Open in Workspace", key=f"open_{sid}", type="primary", width="stretch"):
                    st.session_state.analysis_result = analysis or None
                    st.session_state.gap_report = gaps or None
                    if graph and graph.get("nodes"):
                        st.session_state.knowledge_graph_data = graph
                    elif analysis:
                        pseudo_docs = [
                            {
                                "filename": d.get("filename", name),
                                "title": d.get("filename", name),
                                "page_count": 0,
                                "word_count": 0,
                                "full_text": "",
                            }
                            for d, name in zip(
                                analysis.get("document_analyses", []),
                                doc_names or [x.get("filename", "") for x in analysis.get("document_analyses", [])],
                            )
                        ] or [
                            {"filename": n, "title": n, "page_count": 0, "word_count": 0, "full_text": ""}
                            for n in doc_names
                        ]
                        st.session_state.knowledge_graph_data = build_structural_graph_data(
                            pseudo_docs,
                            analysis_result=analysis,
                            gap_report=gaps or None,
                        )
                    else:
                        st.session_state.knowledge_graph_data = None

                    st.session_state.current_session_id = sid
                    st.session_state.active_history_session = {
                        "session_id": sid,
                        "research_topic": topic,
                        "created_at": created_at,
                        "document_names": doc_names,
                    }
                    st.switch_page("pages/knowledge_analysis.py")

            with act2:
                md_export = format_analysis_markdown(analysis, gaps or None)
                st.download_button(
                    "Download (.md)",
                    data=md_export,
                    file_name=f"{sid.lower()}.md",
                    mime="text/markdown",
                    key=f"dl_{sid}",
                    width="stretch",
                )

            with act3:
                if st.button("Delete", key=f"del_{sid}", width="stretch"):
                    try:
                        delete_research_session(sid, user_id=active_user_id)
                        if st.session_state.get("current_session_id") == sid:
                            st.session_state.current_session_id = None
                        if (
                            st.session_state.get("active_history_session") or {}
                        ).get("session_id") == sid:
                            st.session_state.active_history_session = None
                        st.rerun()
                    except HistoryStorageError as exc:
                        st.error(str(exc))

            with st.expander(f"Preview Saved Results ({sid})", expanded=False):
                if analysis.get("corpus_overview"):
                    st.markdown(f"**Overview:** {analysis['corpus_overview']}")
                if concepts:
                    st.markdown("**Common Concepts:**")
                    for c in concepts[:6]:
                        st.markdown(
                            f"- **{c.get('concept', '')}** ({', '.join(c.get('documents', []))}): "
                            f"{c.get('explanation', '')}"
                        )
                if gaps.get("methodological_gaps"):
                    st.markdown("**Research Gaps:**")
                    for g in gaps["methodological_gaps"][:5]:
                        st.markdown(
                            f"- **[{g.get('severity', 'Medium')}] {g.get('title', '')}**: "
                            f"{g.get('description', '')}"
                        )

            st.write("")

with tab_ambiguity:
    a_col1, a_col2, a_col3 = st.columns([2.2, 1.3, 1.3])
    with a_col1:
        amb_search = st.text_input(
            "Search ambiguity records",
            placeholder="Search original question, resolved question, or document...",
            key="amb_hist_search",
        )
    with a_col2:
        amb_cls_filter = st.selectbox(
            "Classification",
            options=["All", "Clear", "Ambiguous", "Insufficient Context"],
            key="amb_hist_cls",
        )
    with a_col3:
        amb_date_preset = st.selectbox(
            "Date filter",
            options=["All dates", "Today", "Last 7 days", "Last 30 days", "Custom range"],
            key="amb_hist_date_preset",
        )

    amb_custom_dates = None
    if amb_date_preset == "Custom range":
        today = date.today()
        amb_custom_dates = st.date_input(
            "Ambiguity date range",
            value=(today - timedelta(days=30), today),
            key="amb_hist_custom_dates",
        )

    amb_from, amb_to = _resolve_date_bounds(amb_date_preset, amb_custom_dates)

    try:
        all_amb = list_ambiguity_records(user_id=active_user_id)
        filtered_amb = list_ambiguity_records(
            search_query=amb_search,
            classification_filter=amb_cls_filter,
            date_from=amb_from,
            date_to=amb_to,
            user_id=active_user_id,
        )
    except HistoryStorageError as exc:
        st.error(str(exc))
        st.stop()

    ak1, ak2, ak3 = st.columns(3)
    with ak1:
        st.metric("Saved Resolutions", len(all_amb))
    with ak2:
        st.metric("Matching Records", len(filtered_amb))
    with ak3:
        clarified_cnt = sum(1 for r in filtered_amb if r.get("classification") != "Clear")
        st.metric("Clarified Questions", clarified_cnt)

    st.write("")

    if not all_amb:
        with st.container(border=True):
            st.markdown("**No Ambiguity Records Yet**")
            st.write(
                "Ask questions in **Ambiguity Detector**. Every classification, clarification, "
                "rewritten question, and final answer is saved automatically to your account."
            )
    elif not filtered_amb:
        st.info("No ambiguity records match your current filters.")
    else:
        for rec in filtered_amb:
            rid = rec["record_id"]
            cls_label = rec.get("classification", "Clear")
            badge_col = (
                "green" if cls_label == "Clear"
                else "orange" if cls_label == "Ambiguous"
                else "red"
            )
            with st.container(border=True):
                with st.container(horizontal=True):
                    st.markdown(f"**{rec['original_question']}**")
                    st.badge(cls_label, color=badge_col)
                st.markdown(f"**Resolved Question:** {rec['resolved_question']}")
                st.caption(f"Record ID: `{rid}` · Saved: {rec.get('created_at', '')}")
                doc_names_list = rec.get("document_names", [])[:6]
                if doc_names_list:
                    with st.container(horizontal=True):
                        for doc_n in doc_names_list:
                            st.badge(doc_n, color="blue")

            c_det, c_del = st.columns([4.5, 1])
            with c_det:
                with st.expander(f"View Full Resolution Details ({rid})", expanded=False):
                    if rec.get("unclear_phrases"):
                        st.markdown(f"**Unclear Phrases:** {', '.join(rec['unclear_phrases'])}")
                    if rec.get("ambiguity_reason"):
                        st.markdown(f"**Reason:** {rec['ambiguity_reason']}")
                    if rec.get("user_clarification"):
                        st.markdown(f"**User Clarification:** {rec['user_clarification']}")
                    st.markdown(f"**Final Answer:**\n\n{rec.get('final_answer', '')}")
                    if rec.get("citations"):
                        st.markdown("**Citations:**")
                        for cit in rec["citations"]:
                            st.markdown(
                                f"- `{cit.get('filename', '')}` (Page {cit.get('page_number', '?')}): "
                                f"{cit.get('evidence', '')}"
                            )
            with c_del:
                if st.button("Delete", key=f"del_amb_{rid}", width="stretch"):
                    try:
                        delete_ambiguity_record(rid, user_id=active_user_id)
                        st.rerun()
                    except HistoryStorageError as exc:
                        st.error(str(exc))
