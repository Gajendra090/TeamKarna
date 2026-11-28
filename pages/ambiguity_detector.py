"""Ambiguity Detector page: Classify unclear questions, clarify interactively, rewrite, and answer."""

from __future__ import annotations

import streamlit as st

from modules.ai_analysis import AIServiceError, DEFAULT_MODEL, is_api_configured
from modules.ambiguity_detector import (
    complete_clarified_question,
    process_initial_question,
)
from modules.history_db import HistoryStorageError, save_ambiguity_record

st.session_state.setdefault("ambiguity_thread", [])
st.session_state.setdefault("ambiguity_pending", None)
st.session_state.setdefault("ambiguity_clarification_log", [])
st.session_state.setdefault("ambiguity_pending_prompt", "")
st.session_state.setdefault("ambiguity_pending_nonce", 0)


def _badge_color(classification: str) -> str:
    norm = (classification or "").strip().lower()
    if norm == "clear":
        return "green"
    if norm == "ambiguous":
        return "orange"
    return "red"


st.markdown(
    '<div class="kx-hero"><h1>AI Ambiguity Detector</h1>'
    '<p>Detects ambiguous or underspecified questions, asks targeted clarifications, '
    'rewrites intent, and answers from your PDFs.</p></div>',
    unsafe_allow_html=True,
)

def main() -> None:
    docs = st.session_state.get("documents", [])
    if not docs:
        st.warning("No PDFs loaded. Upload one or more research PDFs in the Dashboard or sidebar first.")
        st.stop()
        return

    configured, _ = is_api_configured(st.session_state.get("api_key_override"))
    if not configured:
        st.info("Configure `GEMINI_API_KEY` in `.env` (or enter a session key in the sidebar) to use Ambiguity Detector.")

    all_filenames = [d["filename"] for d in docs]
    ctrl1, ctrl2 = st.columns([3.5, 1.2])
    with ctrl1:
        selected_filenames = st.multiselect(
            "Documents in context",
            options=all_filenames,
            default=all_filenames,
            key="amb_doc_scope",
        )
    with ctrl2:
        st.write("")
        if st.button("New Conversation", width="stretch", key="amb_reset_btn"):
            st.session_state.ambiguity_thread = []
            st.session_state.ambiguity_pending = None
            st.session_state.ambiguity_clarification_log = []
            st.session_state.ambiguity_pending_prompt = ""
            st.session_state.ambiguity_pending_nonce += 1
            st.rerun()

    scoped_docs = [d for d in docs if d["filename"] in selected_filenames]
    if not scoped_docs:
        st.warning("Select at least one document in context above.")
        st.stop()
        return

    # Starter prompts for quick testing across Clear / Ambiguous / Insufficient Context
    if not st.session_state.ambiguity_thread and not st.session_state.ambiguity_pending:
        st.caption("Try example questions:")
        s1, s2, s3 = st.columns(3)
        samples = [
            ("Clear example", f"Summarize the core topics covered in {scoped_docs[0]['filename']}."),
            ("Ambiguous example", "Which approach is better across these documents?"),
            ("Context-needed example", "Will this satisfy my system constraints?"),
        ]
        for idx, (col, (label, sample_q)) in enumerate(zip((s1, s2, s3), samples)):
            with col:
                if st.button(
                    f"{label}: \"{sample_q[:40]}...\"",
                    key=f"amb_starter_{idx}",
                    width="stretch",
                    disabled=not configured,
                ):
                    st.session_state.ambiguity_pending_prompt = sample_q

    # Render completed turns using pure native Streamlit components (zero raw HTML)
    for turn in st.session_state.ambiguity_thread:
        with st.chat_message("user"):
            st.write(turn["original_question"])

        with st.chat_message("assistant"):
            c_info = turn.get("classification_info", {})
            cls_label = c_info.get("classification", "Clear")
            det_conf = c_info.get("confidence", "")

            with st.container(horizontal=True):
                st.badge(cls_label, color=_badge_color(cls_label))
                if det_conf:
                    st.badge(f"Detection: {det_conf}", color="blue")

            if turn.get("user_clarification"):
                with st.container(border=True):
                    st.markdown("**Original Question:**")
                    st.write(turn["original_question"])
                    st.markdown("**Your Clarification:**")
                    st.write(turn["user_clarification"])
                    st.markdown("**Resolved Question:**")
                    st.write(turn["resolved_question"])
            else:
                st.caption(f"Resolved Question: {turn['resolved_question']}")

            st.markdown(turn.get("answer", ""))

            meta_parts = []
            if turn.get("confidence"):
                meta_parts.append(f"Answer Confidence: {turn['confidence']}")
            if turn.get("history_record_id"):
                meta_parts.append(f"Saved in History: `{turn['history_record_id']}`")
            if turn.get("fallback_note"):
                meta_parts.append(turn["fallback_note"])
            if meta_parts:
                st.caption(" · ".join(meta_parts))

            citations = turn.get("citations") or []
            if citations:
                with st.expander(f"Source Citations ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(
                            f"- **`{cit.get('filename', '')}` — Page {cit.get('page_number', '?')}**: "
                            f"{cit.get('evidence', '')}"
                        )

    # Interactive clarification panel when question is Ambiguous or Insufficient Context
    pending = st.session_state.get("ambiguity_pending")
    if pending:
        pending_id = pending.get("pending_id", 0)
        c_info = pending["classification_info"]
        cls_label = c_info.get("classification", "Ambiguous")
        det_conf = c_info.get("confidence", "")
        phrases = c_info.get("unclear_phrases") or []
        reason = c_info.get("ambiguity_reason") or ""
        interps = c_info.get("interpretations") or []
        missing = c_info.get("missing_information") or []
        clar_prompt = c_info.get("clarifying_prompt") or "Please clarify your question."

        with st.chat_message("user"):
            st.write(pending["original_question"])

        with st.chat_message("assistant"):
            with st.container(border=True):
                with st.container(horizontal=True):
                    st.markdown("**Clarification Needed**")
                    st.badge(cls_label, color=_badge_color(cls_label))
                    if det_conf:
                        st.badge(f"Confidence: {det_conf}", color="blue")

                st.markdown("**Original Question:**")
                st.write(pending["original_question"])

                if reason:
                    st.markdown("**Why clarification is needed:**")
                    st.write(reason)

                if phrases:
                    st.markdown("**Unclear words or phrases:**")
                    with st.container(horizontal=True):
                        for phrase in phrases:
                            st.badge(phrase, color="yellow")

                if missing:
                    st.markdown("**Missing context or details:**")
                    for item in missing:
                        st.markdown(f"- {item}")

                if clar_prompt:
                    st.info(clar_prompt)

            chosen_clarification = ""
            if interps:
                radio_options = list(interps) + ["Other (explain my intended meaning below)"]
                selected_option = st.radio(
                    "Select an interpretation",
                    options=radio_options,
                    key=f"amb_interp_radio_{pending_id}",
                )
                custom_text = st.text_input(
                    "Custom clarification or additional details",
                    placeholder="Optional if you selected an option above, required for 'Other'",
                    key=f"amb_custom_clar_{pending_id}",
                )
                if selected_option == "Other (explain my intended meaning below)":
                    chosen_clarification = custom_text.strip()
                else:
                    chosen_clarification = (
                        f"{selected_option}. {custom_text.strip()}".strip()
                        if custom_text.strip()
                        else selected_option
                    )
            else:
                chosen_clarification = st.text_input(
                    "Provide the missing context or details",
                    placeholder="e.g., Target workload, specific algorithm/document, or constraints...",
                    key=f"amb_missing_ctx_{pending_id}",
                ).strip()

            b1, b2 = st.columns([2, 1])
            with b1:
                resolve_clicked = st.button(
                    "Resolve & Generate Answer",
                    type="primary",
                    width="stretch",
                    key=f"amb_resolve_btn_{pending_id}",
                    disabled=not configured,
                )
            with b2:
                cancel_clicked = st.button(
                    "Cancel Question",
                    width="stretch",
                    key=f"amb_cancel_btn_{pending_id}",
                )

            if cancel_clicked:
                st.session_state.ambiguity_pending = None
                st.rerun()

            if resolve_clicked:
                if not chosen_clarification:
                    st.error("Please select an interpretation or enter your clarification first.")
                else:
                    with st.spinner("Rewriting question and generating grounded answer..."):
                        try:
                            resolved_res = complete_clarified_question(
                                original_question=pending["original_question"],
                                user_clarification=chosen_clarification,
                                classification_info=c_info,
                                documents=scoped_docs,
                                chat_history=st.session_state.ambiguity_thread,
                                model=st.session_state.get("selected_model", DEFAULT_MODEL),
                                api_key=st.session_state.get("api_key_override", ""),
                                allow_fallback=st.session_state.get("auto_fallback", True),
                            )
                            try:
                                rec_id = save_ambiguity_record(
                                    documents=scoped_docs,
                                    original_question=resolved_res["original_question"],
                                    classification_info=c_info,
                                    user_clarification=resolved_res["user_clarification"],
                                    resolved_question=resolved_res["resolved_question"],
                                    final_answer=resolved_res["answer"],
                                    citations=resolved_res.get("citations", []),
                                    model_used=resolved_res.get("model_used", ""),
                                    user_id=(st.session_state.get("auth_user") or {}).get("user_id"),
                                )
                            except HistoryStorageError:
                                rec_id = ""

                            st.session_state.ambiguity_clarification_log.append(
                                {
                                    "clarifying_prompt": clar_prompt,
                                    "user_clarification": chosen_clarification,
                                }
                            )
                            st.session_state.ambiguity_thread.append(
                                {
                                    "role": "user",
                                    **resolved_res,
                                    "history_record_id": rec_id,
                                }
                            )
                            st.session_state.ambiguity_pending = None
                            st.rerun()
                        except AIServiceError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Unexpected resolution error: {exc}")

    chat_disabled = (not configured) or bool(st.session_state.get("ambiguity_pending"))
    placeholder_txt = (
        "Resolve or cancel the clarification above before asking a new question..."
        if st.session_state.get("ambiguity_pending")
        else "Ask a question about your uploaded PDFs (will check for ambiguity first)..."
    )
    user_input = st.chat_input(placeholder_txt, disabled=chat_disabled)
    active_question = user_input or st.session_state.pop("ambiguity_pending_prompt", "")

    if active_question and not st.session_state.get("ambiguity_pending"):
        with st.spinner("Analyzing question clarity and document context..."):
            try:
                outcome = process_initial_question(
                    question=active_question,
                    documents=scoped_docs,
                    chat_history=st.session_state.ambiguity_thread,
                    clarification_history=st.session_state.ambiguity_clarification_log,
                    model=st.session_state.get("selected_model", DEFAULT_MODEL),
                    api_key=st.session_state.get("api_key_override", ""),
                    allow_fallback=st.session_state.get("auto_fallback", True),
                )
                if outcome["status"] == "answered":
                    try:
                        rec_id = save_ambiguity_record(
                            documents=scoped_docs,
                            original_question=outcome["original_question"],
                            classification_info=outcome["classification_info"],
                            user_clarification="",
                            resolved_question=outcome["resolved_question"],
                            final_answer=outcome["answer"],
                            citations=outcome.get("citations", []),
                            model_used=outcome.get("model_used", ""),
                            user_id=(st.session_state.get("auth_user") or {}).get("user_id"),
                        )
                    except HistoryStorageError:
                        rec_id = ""
                    st.session_state.ambiguity_thread.append(
                        {
                            "role": "user",
                            **outcome,
                            "history_record_id": rec_id,
                        }
                    )
                else:
                    st.session_state.ambiguity_pending_nonce += 1
                    outcome["pending_id"] = st.session_state.ambiguity_pending_nonce
                    st.session_state.ambiguity_pending = outcome
                st.rerun()
            except AIServiceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected ambiguity analysis error: {exc}")


main()
