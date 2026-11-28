"""My Profile page: Authenticated Google user overview and personal research activity."""

from __future__ import annotations

from html import escape

import streamlit as st

from modules.history_db import HistoryStorageError, get_user_history_summary
from modules.pdf_reader import compute_corpus_stats

st.markdown(
    """
    <div class="kx-hero">
        <h1>My Profile</h1>
        <p>Your Google account details and personal KNOWLEDGE X research activity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

user = st.session_state.get("auth_user")
if not user:
    st.warning("You are not signed in. Click the profile icon at the top of the left sidebar to sign in with Google and view your profile.")
    st.stop()

picture = user.get("picture") or ""
name = user.get("name") or "Google User"
email = user.get("email") or ""
user_id = user.get("user_id") or ""
initials = "".join(part[0].upper() for part in name.split()[:2]) or "U"

avatar_html = (
    f"<img src='{escape(picture)}' alt='Profile photo' "
    f"style='width:68px;height:68px;border-radius:50%;object-fit:cover;border:2px solid rgba(56,189,248,0.5);' />"
    if picture
    else f"<div style='width:68px;height:68px;border-radius:50%;background:linear-gradient(135deg,#0284c7,#4f46e5);"
    f"color:#fff;font-size:1.5rem;font-weight:700;display:flex;align-items:center;justify-content:center;'>"
    f"{escape(initials)}</div>"
)

st.markdown(
    f"""
    <div class="kx-card" style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
        {avatar_html}
        <div>
            <div class="kx-card-title" style="font-size:1.25rem;">{escape(name)}</div>
            <div class="kx-card-body">
                <b>Email:</b> {escape(email)}<br>
                <b>Account ID:</b> <code>{escape(user_id)}</code> &nbsp;&middot;&nbsp;
                <span class="kx-badge kx-badge-ok">{'Authenticated via Email & Password' if user_id.startswith('local:') else 'Authenticated via Google OAuth 2.0'}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    summary = get_user_history_summary(user_id=user_id)
except HistoryStorageError as exc:
    summary = {"research_sessions": 0, "ambiguity_records": 0}
    st.error(str(exc))

docs = st.session_state.get("documents", [])
corpus_stats = compute_corpus_stats(docs)

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Saved Research Analyses", summary["research_sessions"])
with k2:
    st.metric("Saved Ambiguity Resolutions", summary["ambiguity_records"])
with k3:
    st.metric("Active Workspace PDFs", corpus_stats["document_count"])

st.write("")
a1, a2 = st.columns(2)
with a1:
    if st.button("Open My Research History", type="primary", width="stretch"):
        st.switch_page("pages/research_history.py")
with a2:
    if st.button("Back to Dashboard", width="stretch"):
        st.switch_page("pages/dashboard.py")
