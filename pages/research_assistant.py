"""Research Assistant page: Grounded multi-PDF Q&A with page citations."""

from __future__ import annotations

import streamlit as st

from modules.ai_analysis import AIServiceError, answer_research_question, is_api_configured

st.markdown(
    """
    <div class="kx-hero">
        <h1>Research Assistant</h1>
        <p>Ask questions across your uploaded research PDFs and get grounded answers with page citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

docs = st.session_state.get("documents", [])
if not docs:
    st.warning("No PDFs loaded. Upload one or more research PDFs in the Dashboard or sidebar first.")
    st.stop()

configured, _ = is_api_configured(st.session_state.get("api_key_override"))
if not configured:
    st.info("Configure `GEMINI_API_KEY` in `.env` (or enter a session key in the sidebar) to chat with your PDFs.")

all_filenames = [d["filename"] for d in docs]
ctrl1, ctrl2, ctrl3 = st.columns([3.0, 1.3, 1.0])
with ctrl1:
    selected_filenames = st.multiselect(
        "Documents in context",
        options=all_filenames,
        default=all_filenames,
    )
with ctrl2:
    st.write("")
    use_hybrid_search = st.checkbox(
        "Hybrid Search",
        value=True,
        help="Combines Gemini semantic embeddings with lexical keyword matching (falls back to lexical if unavailable).",
    )
with ctrl3:
    st.write("")
    if st.button("Clear Chat", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()

scoped_docs = [d for d in docs if d["filename"] in selected_filenames]
if not scoped_docs:
    st.warning("Select at least one document in context above.")
    st.stop()

# Quick starter prompts when chat is empty
if not st.session_state.get("chat_history"):
    st.caption("Suggested questions:")
    s1, s2, s3 = st.columns(3)
    starters = [
        "Summarize the core thesis and methods of each uploaded PDF.",
        "What common concepts or techniques appear across these documents?",
        "What limitations or open research gaps do these papers highlight?",
    ]
    for col, starter in zip((s1, s2, s3), starters):
        with col:
            if st.button(starter, width="stretch", disabled=not configured):
                st.session_state.pending_question = starter

# Render existing conversation
for msg_idx, message in enumerate(st.session_state.get("chat_history", [])):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            conf = message.get("confidence", "")
            citations = message.get("citations", [])
            fallback_note = message.get("fallback_note", "")
            rmode = message.get("retrieval_mode", "lexical")
            mode_label = (
                "Hybrid Search (Semantic + Lexical)"
                if rmode == "hybrid"
                else "Semantic Search"
                if rmode == "semantic"
                else "Lexical Fallback"
            )
            meta_parts = []
            if conf:
                meta_parts.append(f"Confidence: {conf}")
            meta_parts.append(f"Retrieval: {mode_label}")
            if fallback_note:
                meta_parts.append(fallback_note)
            st.caption(" · ".join(meta_parts))
            if citations:
                with st.expander(f"Source Citations ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(
                            f"- **`{cit.get('filename', '')}` — Page {cit.get('page_number', '?')}**: "
                            f"{cit.get('evidence', '')}"
                        )
            follow_ups = message.get("follow_ups", [])
            if follow_ups and msg_idx == len(st.session_state.chat_history) - 1:
                st.caption("Follow-up questions:")
                for f_idx, fq in enumerate(follow_ups[:3]):
                    if st.button(fq, key=f"followup_{msg_idx}_{f_idx}", disabled=not configured):
                        st.session_state.pending_question = fq
                        st.rerun()

prompt_input = st.chat_input(
    "Ask a question about your uploaded research PDFs...",
    disabled=not configured,
)

active_question = prompt_input or st.session_state.pop("pending_question", "")
if active_question:
    with st.chat_message("user"):
        st.markdown(active_question)

    history_before = list(st.session_state.get("chat_history", []))
    st.session_state.chat_history.append({"role": "user", "content": active_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and synthesizing answer..."):
            try:
                result = answer_research_question(
                    question=active_question,
                    documents=scoped_docs,
                    chat_history=history_before,
                    model=st.session_state.selected_model,
                    api_key=st.session_state.api_key_override,
                    allow_fallback=st.session_state.get("auto_fallback", True),
                    use_semantic=use_hybrid_search,
                )
                answer_text = result.get("answer", "")
                citations = result.get("citations", [])
                confidence = result.get("confidence", "Medium")
                follow_ups = result.get("follow_up_questions", [])
                retrieval_mode = result.get("_retrieval_mode", "lexical")
                mode_label = (
                    "Hybrid Search (Semantic + Lexical)"
                    if retrieval_mode == "hybrid"
                    else "Semantic Search"
                    if retrieval_mode == "semantic"
                    else "Lexical Fallback"
                )

                st.markdown(answer_text)
                st.caption(f"Confidence: {confidence} · Retrieval: {mode_label}")
                if citations:
                    with st.expander(f"Source Citations ({len(citations)})", expanded=True):
                        for cit in citations:
                            st.markdown(
                                f"- **`{cit.get('filename', '')}` — Page {cit.get('page_number', '?')}**: "
                                f"{cit.get('evidence', '')}"
                            )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                        "confidence": confidence,
                        "follow_ups": follow_ups,
                        "fallback_note": result.get("_fallback_note", ""),
                        "retrieval_mode": retrieval_mode,
                    }
                )
                st.rerun()
            except AIServiceError as exc:
                st.error(str(exc))
                st.session_state.chat_history.pop()
            except Exception as exc:
                st.error(f"Unexpected Q&A error: {exc}")
                st.session_state.chat_history.pop()
