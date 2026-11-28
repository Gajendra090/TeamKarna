"""KNOWLEDGE X — AI-Powered Research Knowledge Assistant entrypoint."""

from __future__ import annotations

from html import escape
from pathlib import Path
import time

import streamlit as st
import streamlit.components.v1 as components

from modules.ai_analysis import (
    AIServiceError,
    DEFAULT_MODEL,
    RECOMMENDED_MODELS,
    is_api_configured,
    load_env_config,
    test_gemini_connection,
)
from modules.auth import (
    AuthenticationError,
    SESSION_COOKIE_NAME,
    authenticate_local_user,
    complete_google_oauth_callback,
    cookie_sync_html,
    create_google_authorization_url,
    get_user_uploads_dir,
    init_auth_db,
    is_google_oauth_configured,
    register_local_user,
    revoke_auth_session_cookie,
    verify_auth_session_cookie,
)
from modules.history_db import init_history_db
from modules.pdf_reader import (
    clear_all_uploads,
    compute_corpus_stats,
    load_existing_uploads,
    process_uploaded_files,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
STYLE_PATH = BASE_DIR / "assets" / "style.css"

st.set_page_config(
    page_title="KNOWLEDGE X",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_styles() -> None:
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _reset_workspace_for_user(user_info: dict | None) -> None:
    """Reload workspace documents cleanly for the active user without cross-user leakage."""
    target_dir = get_user_uploads_dir(UPLOADS_DIR, user_info)
    disk_docs, disk_errors = load_existing_uploads(target_dir)
    st.session_state.documents = disk_docs
    st.session_state.upload_errors = disk_errors
    st.session_state.upload_notices = []
    st.session_state.analysis_result = None
    st.session_state.gap_report = None
    st.session_state.knowledge_graph_data = None
    st.session_state.multi_llm_result = None
    st.session_state.chat_history = []
    st.session_state.current_session_id = None
    st.session_state.active_history_session = None
    st.session_state.scanned_queue = []
    st.session_state.last_scan_report = []
    st.session_state.ambiguity_thread = []
    st.session_state.ambiguity_pending = None
    st.session_state.ambiguity_clarification_log = []
    st.session_state.ambiguity_pending_prompt = ""
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
    st.session_state.scan_uploader_nonce = st.session_state.get("scan_uploader_nonce", 0) + 1


def _init_session_state() -> None:
    env_cfg = load_env_config(BASE_DIR / ".env")
    init_history_db()
    init_auth_db()
    defaults = {
        "documents": [],
        "analysis_result": None,
        "gap_report": None,
        "knowledge_graph_data": None,
        "multi_llm_result": None,
        "chat_history": [],
        "api_key_override": "",
        "selected_model": env_cfg["model"] or DEFAULT_MODEL,
        "auto_fallback": True,
        "upload_errors": [],
        "upload_notices": [],
        "uploader_nonce": 0,
        "disk_bootstrapped": False,
        "pending_question": "",
        "current_session_id": None,
        "active_history_session": None,
        "scanned_queue": [],
        "last_scan_report": [],
        "scan_uploader_nonce": 0,
        "ambiguity_thread": [],
        "ambiguity_pending": None,
        "ambiguity_clarification_log": [],
        "ambiguity_pending_prompt": "",
        "ambiguity_pending_nonce": 0,
        "auth_user": None,
        "auth_error": "",
        "auth_notice": "",
        "_pending_cookie_action": None,
        "_pending_oauth_url": "",
        "_pending_oauth_ts": 0.0,
        "_auth_cookie_checked": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _handle_google_auth_lifecycle() -> None:
    """Handle Google OAuth callback query parameters and persistent session cookie restoration."""
    # 1. Process OAuth callback query params if returning from Google login
    qp = st.query_params
    has_code = "code" in qp
    has_error = "error" in qp

    if has_code or has_error:
        code = str(qp.get("code", "") or "")
        state = str(qp.get("state", "") or "")
        err = str(qp.get("error", "") or "")
        err_desc = str(qp.get("error_description", "") or "")
        st.query_params.clear()
        st.session_state._pending_oauth_url = ""

        try:
            user_info, signed_cookie = complete_google_oauth_callback(
                code=code,
                state=state,
                error=err,
                error_description=err_desc,
                env_path=BASE_DIR / ".env",
            )
            st.session_state.auth_user = user_info
            st.session_state.auth_error = ""
            st.session_state.auth_notice = f"Signed in as {user_info['name']}."
            st.session_state._pending_cookie_action = ("set", signed_cookie)
            st.session_state._auth_cookie_checked = True
            _reset_workspace_for_user(user_info)
            st.session_state.disk_bootstrapped = True
        except AuthenticationError as exc:
            st.session_state.auth_error = str(exc)
            st.session_state.auth_notice = ""
        except Exception as exc:
            st.session_state.auth_error = f"Unexpected Google authentication error: {exc}"
            st.session_state.auth_notice = ""

    # 2. Restore persistent session from signed browser cookie on initial load/refresh
    if not st.session_state.get("auth_user") and not st.session_state.get("_auth_cookie_checked"):
        st.session_state._auth_cookie_checked = True
        try:
            raw_cookie = st.context.cookies.get(SESSION_COOKIE_NAME, "")
        except Exception:
            raw_cookie = ""
        if raw_cookie:
            restored = verify_auth_session_cookie(raw_cookie, env_path=BASE_DIR / ".env")
            if restored:
                st.session_state.auth_user = restored
                _reset_workspace_for_user(restored)
                st.session_state.disk_bootstrapped = True
            else:
                st.session_state._pending_cookie_action = ("clear", "")

    # 3. Bootstrap active user's disk uploads on first load
    if not st.session_state.disk_bootstrapped:
        target_dir = get_user_uploads_dir(UPLOADS_DIR, st.session_state.get("auth_user"))
        disk_docs, disk_errors = load_existing_uploads(target_dir)
        if disk_docs:
            st.session_state.documents = disk_docs
        if disk_errors:
            st.session_state.upload_errors = disk_errors
        st.session_state.disk_bootstrapped = True


def _perform_logout() -> None:
    """Revoke server-side session, clear browser cookie, and wipe user-scoped workspace state."""
    try:
        raw_cookie = st.context.cookies.get(SESSION_COOKIE_NAME, "")
    except Exception:
        raw_cookie = ""
    if raw_cookie:
        revoke_auth_session_cookie(raw_cookie, env_path=BASE_DIR / ".env")

    st.session_state.auth_user = None
    st.session_state.auth_error = ""
    st.session_state.auth_notice = "You have signed out."
    st.session_state._pending_cookie_action = ("clear", "")
    st.session_state._pending_oauth_url = ""
    st.session_state._pending_oauth_ts = 0.0
    _reset_workspace_for_user(None)
    st.session_state.disk_bootstrapped = True


_PERSON_SILHOUETTE_SVG = (
    "<svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
    "<path d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'/>"
    "<circle cx='12' cy='7' r='4'/>"
    "</svg>"
)


def _get_oauth_state() -> tuple[bool, str, str]:
    """Return (oauth_ready, status_message, active_pkce_auth_url) using modules/auth.py."""
    oauth_ready, oauth_msg = is_google_oauth_configured(BASE_DIR / ".env")
    if not oauth_ready:
        return False, oauth_msg, ""
    now = time.time()
    if not st.session_state.get("_pending_oauth_url") or (
        now - float(st.session_state.get("_pending_oauth_ts") or 0.0) > 480
    ):
        try:
            auth_url, _ = create_google_authorization_url(env_path=BASE_DIR / ".env")
            st.session_state._pending_oauth_url = auth_url
            st.session_state._pending_oauth_ts = now
        except AuthenticationError as exc:
            st.session_state.auth_error = str(exc)
    return True, oauth_msg, st.session_state.get("_pending_oauth_url", "")


def _build_avatar_html(
    user_info: dict,
    img_cls: str = "kx-sidebar-avatar-img",
    fallback_cls: str = "kx-sidebar-avatar-fallback",
) -> str:
    """Build safe HTML for the user's Google profile picture or initials fallback."""
    name = user_info.get("name") or "Google User"
    picture = user_info.get("picture") or ""
    initials = "".join(part[0].upper() for part in name.split()[:2]) or "U"
    if picture:
        return (
            f"<img class='{img_cls}' src='{escape(picture)}' alt='{escape(name)}' "
            f"referrerpolicy='no-referrer' />"
        )
    return f"<div class='{fallback_cls}'>{escape(initials)}</div>"


def _render_auth_status_banner() -> None:
    """Display transient sign-in/sign-out or OAuth callback errors cleanly."""
    if st.session_state.get("auth_error"):
        st.error(st.session_state.auth_error)
        st.session_state.auth_error = ""
    elif st.session_state.get("auth_notice"):
        st.success(st.session_state.auth_notice)
        st.session_state.auth_notice = ""


def _flush_pending_cookie() -> None:
    """Emit browser auth-cookie update script before active page execution."""
    pending_cookie = st.session_state.get("_pending_cookie_action")
    st.session_state._pending_cookie_action = None
    if pending_cookie:
        action, cookie_val = pending_cookie
        html_js = cookie_sync_html(action, cookie_val)
        if html_js:
            with st.container(key="kx_cookie_sync"):
                components.html(html_js, height=0)


def _invalidate_derived_state() -> None:
    st.session_state.analysis_result = None
    st.session_state.gap_report = None
    st.session_state.knowledge_graph_data = None
    st.session_state.multi_llm_result = None
    st.session_state.current_session_id = None
    st.session_state.active_history_session = None


def handle_uploaded_pdfs(uploaded_files) -> bool:
    """Ingest uploaded PDFs into session state and user-scoped uploads directory."""
    if not uploaded_files:
        return False
    target_dir = get_user_uploads_dir(UPLOADS_DIR, st.session_state.get("auth_user"))
    updated_docs, added_names, errors = process_uploaded_files(
        uploaded_files=uploaded_files,
        existing_docs=st.session_state.documents,
        save_dir=target_dir,
    )
    if added_names:
        st.session_state.documents = updated_docs
        _invalidate_derived_state()
        st.session_state.upload_notices = [
            f"Loaded {len(added_names)} PDF(s): {', '.join(added_names)}"
        ]
    if errors:
        st.session_state.upload_errors = errors
    return bool(added_names or errors)


def _apply_authenticated_user(user_info: dict, signed_cookie: str, notice: str) -> None:
    """Activate an authenticated user session, persist cookie, and scope workspace."""
    st.session_state.auth_user = user_info
    st.session_state.auth_error = ""
    st.session_state.auth_notice = notice
    st.session_state._pending_cookie_action = ("set", signed_cookie)
    st.session_state._auth_cookie_checked = True
    _reset_workspace_for_user(user_info)
    st.session_state.disk_bootstrapped = True


def _render_sidebar_profile_avatar() -> None:
    """Render a compact circular profile trigger at the very top of the sidebar, above Dashboard."""
    user = st.session_state.get("auth_user")
    with st.container(key="kx_sidebar_auth"):
        if not user:
            st.markdown(
                f"<div class='kx-sidebar-avatar-visual kx-sidebar-avatar-default' "
                f"title='Account sign in' aria-hidden='true'>{_PERSON_SILHOUETTE_SVG}</div>",
                unsafe_allow_html=True,
            )
            with st.popover(
                " ",
                width="content",
                help="Sign in or Sign up",
                key="kx_sidebar_auth_popover",
            ):
                tab_signin, tab_signup = st.tabs(["Sign In", "Sign Up"])
                with tab_signin:
                    with st.form("kx_local_login_form", clear_on_submit=False):
                        login_email = st.text_input("Email", placeholder="you@example.com", key="login_email_input")
                        login_password = st.text_input("Password", type="password", key="login_pw_input")
                        login_submit = st.form_submit_button("Sign In", type="primary", width="stretch")
                    if login_submit:
                        try:
                            user_info, signed_cookie = authenticate_local_user(
                                email=login_email,
                                password=login_password,
                                env_path=BASE_DIR / ".env",
                            )
                            _apply_authenticated_user(
                                user_info,
                                signed_cookie,
                                f"Signed in as {user_info['name']}.",
                            )
                            st.rerun()
                        except AuthenticationError as exc:
                            st.error(str(exc))

                    st.divider()
                    oauth_ready, oauth_msg, auth_url = _get_oauth_state()
                    if oauth_ready and auth_url:
                        st.markdown(
                            f"<a class='kx-sidebar-oauth-btn' href='{escape(auth_url)}' target='_self'>"
                            f"Sign in with Google</a>",
                            unsafe_allow_html=True,
                        )
                    else:
                        if st.button(
                            "Sign in with Google",
                            width="stretch",
                            key="sidebar_oauth_signin_btn",
                        ):
                            st.warning(oauth_msg)
                            st.caption(
                                "Configure `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REDIRECT_URI` "
                                "in `.env` or `.streamlit/secrets.toml` to start Google OAuth login."
                            )

                with tab_signup:
                    with st.form("kx_local_signup_form", clear_on_submit=False):
                        signup_name = st.text_input("Full Name", placeholder="Your name", key="signup_name_input")
                        signup_email = st.text_input("Email", placeholder="you@example.com", key="signup_email_input")
                        signup_pw = st.text_input(
                            "Password",
                            type="password",
                            help="Minimum 8 characters",
                            key="signup_pw_input",
                        )
                        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm_input")
                        signup_submit = st.form_submit_button("Create Account", type="primary", width="stretch")
                    if signup_submit:
                        try:
                            user_info, signed_cookie = register_local_user(
                                name=signup_name,
                                email=signup_email,
                                password=signup_pw,
                                confirm_password=signup_confirm,
                                env_path=BASE_DIR / ".env",
                            )
                            _apply_authenticated_user(
                                user_info,
                                signed_cookie,
                                f"Account created! Welcome, {user_info['name']}.",
                            )
                            st.rerun()
                        except AuthenticationError as exc:
                            st.error(str(exc))
        else:
            name = user.get("name") or "Google User"
            email = user.get("email") or ""
            avatar_html = _build_avatar_html(
                user,
                img_cls="kx-sidebar-avatar-img",
                fallback_cls="kx-sidebar-avatar-fallback",
            )
            st.markdown(
                f"<div class='kx-sidebar-avatar-visual' title='{escape(name)}' aria-hidden='true'>"
                f"{avatar_html}</div>",
                unsafe_allow_html=True,
            )
            with st.popover(
                " ",
                width="content",
                help=f"{name} ({email})",
                key="kx_sidebar_auth_popover",
            ):
                st.markdown(f"**{name}**")
                if email:
                    st.caption(email)
                st.divider()
                if st.button(
                    "My Profile",
                    icon=":material/person:",
                    width="stretch",
                    key="sidebar_my_profile",
                ):
                    st.switch_page("pages/profile.py")
                st.button(
                    "Logout",
                    icon=":material/logout:",
                    width="stretch",
                    key="sidebar_logout",
                    on_click=_perform_logout,
                )


def _render_sidebar() -> None:
    with st.sidebar:
        _render_sidebar_profile_avatar()
        st.markdown(
            """
            <div class="kx-brand">
                <div class="kx-brand-logo" aria-label="KNOWLEDGE X">
                    <svg class="kx-brand-svg" viewBox="0 0 360 76" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="KNOWLEDGE X">
                        <defs>
                            <linearGradient id="kxWordGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#ffffff" />
                                <stop offset="100%" stop-color="#e2e8f0" />
                            </linearGradient>
                            <linearGradient id="kxLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#a855f7" stop-opacity="0.15" />
                                <stop offset="28%" stop-color="#a855f7" stop-opacity="0.95" />
                                <stop offset="100%" stop-color="#7c3aed" stop-opacity="1" />
                            </linearGradient>
                            <linearGradient id="kxXGrad" x1="232" y1="8" x2="342" y2="68" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stop-color="#d946ef" />
                                <stop offset="45%" stop-color="#8b5cf6" />
                                <stop offset="100%" stop-color="#0ea5e9" />
                            </linearGradient>
                            <linearGradient id="kxXBladeGrad" x1="230" y1="70" x2="356" y2="4" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stop-color="#7c3aed" />
                                <stop offset="55%" stop-color="#9333ea" />
                                <stop offset="100%" stop-color="#38bdf8" />
                            </linearGradient>
                        </defs>
                        <text
                            x="6"
                            y="45"
                            fill="url(#kxWordGrad)"
                            font-family="'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
                            font-size="31"
                            font-weight="800"
                            font-style="italic"
                            letter-spacing="1.8"
                        >KNOWLEDGE</text>
                        <rect x="6" y="53" width="244" height="2" rx="1" fill="url(#kxLineGrad)" />
                        <path
                            d="M244 15 H271 L286 34 L274 42 L244 15 Z M289 40 L303 32 L322 59 H296 L289 40 Z"
                            fill="url(#kxXGrad)"
                        />
                        <path
                            d="M230 71 L276 37 L356 4 L291 43 L249 71 H230 Z"
                            fill="url(#kxXBladeGrad)"
                        />
                    </svg>
                </div>
                <div class="kx-brand-sub">Rapid Research Internalization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Research PDFs")
        sidebar_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"sidebar_pdf_uploader_{st.session_state.uploader_nonce}",
            help="Uploaded PDFs are validated with PyMuPDF and stored securely.",
        )
        if sidebar_files and handle_uploaded_pdfs(sidebar_files):
            st.session_state.uploader_nonce += 1
            st.rerun()

        for note in st.session_state.get("upload_notices") or []:
            st.success(note)
        for err in st.session_state.get("upload_errors") or []:
            st.error(err)
        st.session_state.upload_notices = []
        st.session_state.upload_errors = []

        stats = compute_corpus_stats(st.session_state.documents)
        if stats["document_count"] > 0:
            st.markdown(
                f"<div class='kx-status-pill'>"
                f"<span><b>{stats['document_count']}</b> PDF(s)</span>"
                f"<span>&middot;</span>"
                f"<span>{stats['total_pages']} pages</span>"
                f"<span>&middot;</span>"
                f"<span>{stats['total_words']:,} words</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Clear All PDFs", width="stretch", key="sidebar_clear_pdfs"):
                target_dir = get_user_uploads_dir(UPLOADS_DIR, st.session_state.get("auth_user"))
                clear_all_uploads(target_dir)
                st.session_state.documents = []
                st.session_state.chat_history = []
                st.session_state.upload_errors = []
                st.session_state.upload_notices = ["Cleared all uploaded PDFs."]
                _invalidate_derived_state()
                st.session_state.uploader_nonce += 1
                st.rerun()
        else:
            st.caption("No PDFs loaded yet.")

        st.divider()
        st.markdown("#### Gemini Configuration")

        configured, status_msg = is_api_configured(st.session_state.api_key_override)
        badge_cls = "kx-badge-ok" if configured else "kx-badge-high"
        st.markdown(
            f"<span class='kx-badge {badge_cls}'>{escape(status_msg)}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        model_choices = list(RECOMMENDED_MODELS)
        current_model = st.session_state.selected_model
        if current_model and current_model not in model_choices:
            model_choices.insert(0, current_model)

        st.session_state.selected_model = st.selectbox(
            "Gemini Model",
            options=model_choices,
            index=model_choices.index(current_model) if current_model in model_choices else 0,
            help="Configured via GEMINI_MODEL in .env or overridden here for this session.",
        )

        st.session_state.auto_fallback = st.checkbox(
            "Auto-fallback on high demand",
            value=st.session_state.get("auto_fallback", True),
            help="If the selected model returns 503/429/404, automatically retry with a healthy Flash model.",
        )

        if st.button("Test Gemini Connection", width="stretch", disabled=not configured):
            with st.spinner("Testing Gemini API..."):
                try:
                    diag = test_gemini_connection(
                        model=st.session_state.selected_model,
                        api_key=st.session_state.api_key_override,
                    )
                    st.success(
                        f"Connected to `{diag['model']}` ({diag['available_count']} models accessible)."
                    )
                except AIServiceError as exc:
                    st.error(str(exc))

        with st.expander("API Key Override (Optional)", expanded=not configured):
            override = st.text_input(
                "Gemini API Key",
                value=st.session_state.api_key_override,
                type="password",
                placeholder="Reads GEMINI_API_KEY from .env by default",
                help="Stored only in session memory. Recommended: set GEMINI_API_KEY in .env.",
            )
            st.session_state.api_key_override = override.strip()


_load_styles()
_init_session_state()
_handle_google_auth_lifecycle()

pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("pages/knowledge_analysis.py", title="Knowledge Analysis", icon=":material/analytics:"),
    st.Page("pages/knowledge_map.py", title="Knowledge Map", icon=":material/hub:"),
    st.Page("pages/research_assistant.py", title="Research Assistant", icon=":material/forum:"),
    st.Page("pages/ambiguity_detector.py", title="Ambiguity Detector", icon=":material/rule:"),
    st.Page("pages/research_history.py", title="Research History", icon=":material/history:"),
    st.Page("pages/profile.py", title="My Profile", icon=":material/person:", visibility="hidden"),
]

navigation = st.navigation(pages, position="sidebar")
_render_sidebar()
_render_auth_status_banner()
_flush_pending_cookie()
navigation.run()
