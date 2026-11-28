"""Dashboard page: Multi-PDF upload, URL link upload, OCR document scan, pipeline runner, and text inspector."""

from __future__ import annotations

import hashlib
from html import escape
from pathlib import Path

import streamlit as st

from modules.ai_analysis import (
    AIServiceError,
    DEFAULT_EMBEDDING_MODEL,
    RECOMMENDED_MODELS,
    analyze_documents,
    is_api_configured,
)
from modules.auth import get_user_uploads_dir
from modules.camera_component import render_persistent_camera
from modules.history_db import HistoryStorageError, get_chunk_embeddings, save_research_session
from modules.knowledge_map import build_structural_graph_data
from modules.multi_llm_pipeline import WorkerModelConfig, run_multi_llm_pipeline
from modules.ocr_scanner import ScanProcessingError, process_scanned_pages
from modules.pdf_reader import (
    PDFProcessingError,
    compute_corpus_stats,
    fetch_pdf_from_url,
    process_uploaded_files,
    remove_document,
)
from modules.research_gap import detect_research_gaps
from modules.semantic_search import index_document_embeddings

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"


def _active_uploads_dir() -> Path:
    return get_user_uploads_dir(UPLOADS_DIR, st.session_state.get("auth_user"))


def _active_user_id() -> str | None:
    user = st.session_state.get("auth_user") or {}
    return user.get("user_id")

st.session_state.setdefault("uploader_nonce", 0)
st.session_state.setdefault("scan_uploader_nonce", 0)
st.session_state.setdefault("pending_cam_capture", None)
st.session_state.setdefault("documents", [])
st.session_state.setdefault("scanned_queue", [])
st.session_state.setdefault("last_scan_report", [])


def _reset_analysis_after_doc_change() -> None:
    st.session_state.analysis_result = None
    st.session_state.gap_report = None
    st.session_state.knowledge_graph_data = None
    st.session_state.multi_llm_result = None
    st.session_state.current_session_id = None
    st.session_state.active_history_session = None


st.markdown(
    """
    <div class="kx-hero">
        <h1>KNOWLEDGE X Dashboard</h1>
        <p>Upload PDFs, import from public links, or scan multi-page documents with OCR to synthesize research.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

docs = st.session_state.get("documents", [])
stats = compute_corpus_stats(docs)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Uploaded PDFs", stats["document_count"])
with col2:
    st.metric("Total Pages", stats["total_pages"])
with col3:
    st.metric("Extracted Words", f"{stats['total_words']:,}")
with col4:
    ai_ready = "Completed" if st.session_state.get("analysis_result") else "Pending"
    st.metric("Knowledge Analysis", ai_ready)

st.write("")

configured, _ = is_api_configured(st.session_state.get("api_key_override"))

# Unified Document Ingestion Hub
st.subheader("Add Research Documents")
tab_pdf, tab_link, tab_scan = st.tabs(
    ["Upload PDF", "Upload Using Link", "Scan Documents"],
    key="dashboard_doc_tab",
    on_change="rerun",
)

with tab_pdf:
    uploaded_main = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"dashboard_pdf_uploader_{st.session_state.get('uploader_nonce', 0)}",
        help="Select multiple PDF files to extract text and analyze together.",
    )
    if uploaded_main:
        updated_docs, added_names, errors = process_uploaded_files(
            uploaded_files=uploaded_main,
            existing_docs=st.session_state.documents,
            save_dir=_active_uploads_dir(),
        )
        if added_names:
            st.session_state.documents = updated_docs
            _reset_analysis_after_doc_change()
            st.session_state.upload_notices = [
                f"Loaded {len(added_names)} PDF(s): {', '.join(added_names)}"
            ]
        if errors:
            st.session_state.upload_errors = errors
        if added_names or errors:
            st.session_state.uploader_nonce += 1
            st.rerun()

with tab_link:
    st.caption(
        "Download a publicly accessible PDF directly into your workspace. "
        "Localhost, private networks, and non-PDF links are blocked for security."
    )
    link_col1, link_col2 = st.columns([3.5, 1.3])
    with link_col1:
        pdf_url = st.text_input(
            "Public PDF URL",
            placeholder="https://arxiv.org/pdf/1706.03762",
            key="pdf_url_input",
            label_visibility="collapsed",
        )
    with link_col2:
        fetch_btn = st.button("Download & Extract", type="primary", width="stretch")

    if fetch_btn:
        with st.status("Downloading and validating remote PDF...", expanded=True) as url_status:
            try:
                st.write("Validating public URL and checking network safety...")
                updated_docs, record, status_msg = fetch_pdf_from_url(
                    url=pdf_url,
                    existing_docs=st.session_state.documents,
                    save_dir=_active_uploads_dir(),
                )
                is_new = len(updated_docs) > len(st.session_state.documents)
                st.session_state.documents = updated_docs
                if is_new:
                    _reset_analysis_after_doc_change()
                url_status.update(label=status_msg, state="complete", expanded=False)
                st.success(status_msg)
                if is_new:
                    st.session_state.upload_notices = [status_msg]
                    st.rerun()
            except PDFProcessingError as exc:
                url_status.update(label="PDF link download failed.", state="error", expanded=False)
                st.error(str(exc))
            except Exception as exc:
                url_status.update(label="Unexpected download error.", state="error", expanded=False)
                st.error(f"Unexpected error downloading PDF: {exc}")

with tab_scan:
    st.caption(
        "Capture document pages with your device camera or upload scanned page images. "
        "Pages are checked for blur, processed via OCR, and saved as a searchable PDF."
    )
    if not configured:
        st.info("Configure `GEMINI_API_KEY` in `.env` (or sidebar override) to run OCR text extraction.")

    scan_left, scan_right = st.columns(2)
    with scan_left:
        try:
            req_host = (st.context.headers.get("host") or "").lower()
            req_proto = (st.context.headers.get("x-forwarded-proto") or "").lower()
        except Exception:
            req_host, req_proto = "", ""
        host_only = req_host.split(":")[0]
        if host_only and host_only not in {"localhost", "127.0.0.1", "[::1]"} and req_proto != "https":
            st.warning(
                "Live camera access requires `https://` or `http://localhost`. "
                "On mobile or LAN HTTP URLs, use **Upload scanned page images** on the right "
                "to take a photo directly with your device camera."
            )

        captured_bytes = render_persistent_camera(key="kx_live_camera")
        if captured_bytes:
            st.session_state.pending_cam_capture = captured_bytes

        pending_bytes = st.session_state.get("pending_cam_capture")
        if pending_bytes:
            cam_hash = hashlib.sha256(pending_bytes).hexdigest()
            st.image(pending_bytes, caption="Captured photo preview", width="stretch")
            cam_btn_col1, cam_btn_col2 = st.columns(2)
            with cam_btn_col1:
                if st.button("Add Captured Photo to Queue", type="primary", width="stretch", key="add_cam_page"):
                    existing_hashes = {item["hash"] for item in st.session_state.scanned_queue}
                    if cam_hash not in existing_hashes:
                        page_num = len(st.session_state.scanned_queue) + 1
                        st.session_state.scanned_queue.append(
                            {
                                "name": f"camera_page_{page_num}.jpg",
                                "bytes": pending_bytes,
                                "hash": cam_hash,
                            }
                        )
                        st.session_state.pending_cam_capture = None
                        st.session_state.upload_notices = [
                            f"Added camera_page_{page_num}.jpg to scan queue."
                        ]
                        st.rerun()
                    else:
                        st.info("This captured photo is already in the scan queue.")
            with cam_btn_col2:
                if st.button("Discard & Capture Another", width="stretch", key="retake_cam_page"):
                    st.session_state.pending_cam_capture = None
                    st.rerun()

    with scan_right:
        scan_uploads = st.file_uploader(
            "Upload scanned page images (multi-page supported)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            accept_multiple_files=True,
            key=f"scan_img_uploader_{st.session_state.get('scan_uploader_nonce', 0)}",
        )
        if scan_uploads:
            existing_hashes = {item["hash"] for item in st.session_state.scanned_queue}
            added_imgs = 0
            for img_file in scan_uploads:
                raw_img = img_file.getvalue()
                img_hash = hashlib.sha256(raw_img).hexdigest()
                if img_hash not in existing_hashes:
                    st.session_state.scanned_queue.append(
                        {
                            "name": img_file.name,
                            "bytes": raw_img,
                            "hash": img_hash,
                        }
                    )
                    existing_hashes.add(img_hash)
                    added_imgs += 1
            if added_imgs:
                st.session_state.scan_uploader_nonce += 1
                st.rerun()

    queue = st.session_state.get("scanned_queue", [])
    if queue:
        st.markdown(f"**Staged Scanned Pages ({len(queue)}):**")
        preview_cols = st.columns(min(4, len(queue)))
        for idx, item in enumerate(queue):
            with preview_cols[idx % len(preview_cols)]:
                st.image(item["bytes"], caption=f"Page {idx + 1}: {item['name']}", width="stretch")
                if st.button("Remove", key=f"rm_scan_{item['hash']}", width="stretch"):
                    st.session_state.scanned_queue = [
                        p for p in queue if p["hash"] != item["hash"]
                    ]
                    st.rerun()

        s_col1, s_col2, s_col3 = st.columns([2.4, 2, 1.2])
        with s_col1:
            scan_doc_name = st.text_input(
                "Output PDF filename",
                value="scanned_document.pdf",
                key="scan_output_filename",
            )
        with s_col2:
            st.write("")
            run_ocr_btn = st.button(
                "Extract Text (OCR) & Convert to PDF",
                type="primary",
                width="stretch",
                disabled=not configured,
            )
        with s_col3:
            st.write("")
            if st.button("Clear Queue", width="stretch", key="clear_scan_queue"):
                st.session_state.scanned_queue = []
                st.rerun()

        if run_ocr_btn:
            with st.status("Scanning pages and running OCR...", expanded=True) as scan_status:
                try:
                    def _on_scan_progress(cur: int, tot: int, msg: str) -> None:
                        st.write(msg)

                    updated_docs, doc_record, reports, scan_warnings = process_scanned_pages(
                        pages_input=queue,
                        output_filename=scan_doc_name,
                        existing_docs=st.session_state.documents,
                        save_dir=_active_uploads_dir(),
                        model=st.session_state.selected_model,
                        api_key=st.session_state.api_key_override,
                        allow_fallback=st.session_state.get("auto_fallback", True),
                        on_progress=_on_scan_progress,
                    )
                    st.session_state.documents = updated_docs
                    st.session_state.last_scan_report = reports
                    st.session_state.scanned_queue = []
                    _reset_analysis_after_doc_change()
                    scan_status.update(
                        label=f"Created searchable PDF '{doc_record['filename']}' ({doc_record['page_count']} pages, {doc_record['word_count']:,} words).",
                        state="complete",
                        expanded=False,
                    )
                    notice = (
                        f"Scanned '{doc_record['filename']}' "
                        f"({doc_record['page_count']} page(s), {doc_record['word_count']:,} words)."
                    )
                    st.session_state.upload_notices = [notice]
                    if scan_warnings:
                        st.session_state.upload_errors = scan_warnings
                    st.rerun()
                except (ScanProcessingError, AIServiceError) as exc:
                    scan_status.update(label="Scan OCR failed.", state="error", expanded=False)
                    st.error(str(exc))
                except Exception as exc:
                    scan_status.update(label="Unexpected scan error.", state="error", expanded=False)
                    st.error(f"Unexpected scanning error: {exc}")

    last_reports = st.session_state.get("last_scan_report") or []
    if last_reports:
        with st.expander("Last Scan OCR Status Report", expanded=True):
            rows_html = "".join(
                f"<tr>"
                f"<td>Page {r['page']}</td>"
                f"<td>{escape(str(r['filename']))}</td>"
                f"<td><span class='kx-badge {'kx-badge-ok' if r['status'] == 'Extracted' else 'kx-badge-high'}'>{escape(str(r['status']))}</span></td>"
                f"<td>{escape(str(r['legibility']))}</td>"
                f"<td>{r['word_count']}</td>"
                f"<td>{escape(str(r.get('issue') or 'OK'))}</td>"
                f"</tr>"
                for r in last_reports
            )
            st.markdown(
                f"""
                <table class="kx-table">
                    <thead><tr><th>Page</th><th>Image</th><th>Status</th><th>Legibility</th><th>Words</th><th>Quality / Note</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

if not docs:
    st.markdown(
        """
        <div class="kx-card">
            <div class="kx-card-title">Getting Started</div>
            <div class="kx-card-body">
                1. Add research documents using <b>Upload PDF</b>, <b>Upload Using Link</b>, or <b>Scan Documents</b> above.<br>
                2. Ensure your <code>GEMINI_API_KEY</code> is configured in <code>.env</code> (or enter a session key in the sidebar).<br>
                3. Open <b>Knowledge Analysis</b> for structured summaries, common concepts, and gap detection.<br>
                4. Browse past runs anytime in <b>Research History</b>, or explore <b>Knowledge Map</b> and <b>Research Assistant</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.divider()

# KnowledgeX 2.0: Semantic Indexing & Multi-LLM Parallel Orchestration
total_chunks_count = sum(len(d.get("chunks") or []) for d in docs)
doc_hashes = [str(d.get("file_hash") or "") for d in docs if d.get("file_hash")]
try:
    indexed_emb_map = get_chunk_embeddings(doc_hashes, DEFAULT_EMBEDDING_MODEL) if doc_hashes else {}
except HistoryStorageError:
    indexed_emb_map = {}

indexed_chunks_count = len(indexed_emb_map)
multi_res_existing = st.session_state.get("multi_llm_result")
multi_status_label = (
    f"Multi-LLM: {multi_res_existing.get('status', 'completed').title()}"
    if multi_res_existing
    else "Multi-LLM: Pending"
)

st.subheader("KnowledgeX 2.0 Indexing & Multi-LLM Pipeline")
st.caption(
    f"Section Chunks Cached: **{total_chunks_count}** · "
    f"Semantic Embeddings Indexed (`{DEFAULT_EMBEDDING_MODEL}`): **{indexed_chunks_count}/{total_chunks_count}** · "
    f"{multi_status_label}"
)

k2_col1, k2_col2 = st.columns(2)
with k2_col1:
    run_index_btn = st.button(
        "Index Semantic Embeddings",
        width="stretch",
        disabled=not configured,
        help="Incrementally generates Gemini embeddings for unindexed PDF chunks (skips already indexed chunks).",
    )
with k2_col2:
    run_multi_llm_btn = st.button(
        "Run Multi-LLM Synthesis (2.0)",
        type="primary",
        width="stretch",
        disabled=not configured,
        help="Runs bounded parallel Workers 1 & 2, Worker 3 Research Intelligence, and Master Aggregator with SQLite caching.",
    )

with st.expander("Multi-LLM Worker Model Configuration (Optional)", expanded=False):
    default_m = st.session_state.get("selected_model", "gemini-3.6-flash")
    mc1, mc2 = st.columns(2)
    with mc1:
        w1_model = st.selectbox(
            "Worker 1 (Document Analysis)",
            options=RECOMMENDED_MODELS,
            index=RECOMMENDED_MODELS.index(default_m) if default_m in RECOMMENDED_MODELS else 0,
            key="kx2_w1_model",
        )
        w3_model = st.selectbox(
            "Worker 3 (Research Intelligence)",
            options=RECOMMENDED_MODELS,
            index=RECOMMENDED_MODELS.index(default_m) if default_m in RECOMMENDED_MODELS else 0,
            key="kx2_w3_model",
        )
    with mc2:
        w2_default = "gemini-3.5-flash-lite" if "gemini-3.5-flash-lite" in RECOMMENDED_MODELS else default_m
        w2_model = st.selectbox(
            "Worker 2 (Section Summarization)",
            options=RECOMMENDED_MODELS,
            index=RECOMMENDED_MODELS.index(w2_default) if w2_default in RECOMMENDED_MODELS else 0,
            key="kx2_w2_model",
        )
        master_model = st.selectbox(
            "Master LLM (Final Aggregator)",
            options=RECOMMENDED_MODELS,
            index=RECOMMENDED_MODELS.index(default_m) if default_m in RECOMMENDED_MODELS else 0,
            key="kx2_master_model",
        )

if run_index_btn:
    with st.spinner("Indexing unindexed PDF chunks with Gemini embeddings..."):
        try:
            idx_summary = index_document_embeddings(
                documents=docs,
                model=DEFAULT_EMBEDDING_MODEL,
                api_key=st.session_state.get("api_key_override"),
            )
            st.success(
                f"Embedding indexing complete ({idx_summary['embedding_model']}): "
                f"{idx_summary['indexed_count']} newly indexed, {idx_summary['cached_count']} reused from SQLite cache."
            )
        except (AIServiceError, HistoryStorageError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected embedding indexing error: {exc}")

if run_multi_llm_btn:
    api_key = st.session_state.get("api_key_override")
    allow_fallback = st.session_state.get("auto_fallback", True)
    cfg = WorkerModelConfig(
        worker1_model=w1_model,
        worker2_model=w2_model,
        worker3_model=w3_model,
        master_model=master_model,
    )
    try:
        with st.status("Running KnowledgeX 2.0 Multi-LLM Pipeline...", expanded=True) as m_status:
            st.write(f"Running Workers 1 & 2 across {len(docs)} PDF(s) (max 3 concurrent) + Worker 3 + Master Aggregator...")
            m_res = run_multi_llm_pipeline(
                documents=docs,
                model_config=cfg,
                api_key=api_key,
                allow_fallback=allow_fallback,
            )
            st.session_state.multi_llm_result = m_res
            if m_res["status"] == "failed":
                m_status.update(label="Multi-LLM pipeline failed for all documents.", state="error", expanded=True)
                for fd in m_res.get("failed_documents", []):
                    st.error(f"{fd['filename']} ({fd['worker']}): {fd['error']}")
            else:
                usage = m_res.get("token_usage") or {}
                label = (
                    f"Multi-LLM synthesis {m_res['status']} · "
                    f"{m_res.get('cache_hits', 0)} cache hit(s) · "
                    f"{usage.get('total_tokens', 0):,} tokens"
                )
                m_status.update(label=label, state="complete", expanded=False)
                if m_res.get("failed_documents"):
                    st.warning(
                        f"Completed with partial failures ({len(m_res['failed_documents'])} worker task(s) failed). "
                        "Re-run to retry failed documents while reusing cached results."
                    )
                    for fd in m_res["failed_documents"]:
                        st.caption(f"- `{fd['filename']}` ({fd['worker']}): {fd['error']}")
                else:
                    st.success(
                        f"Multi-LLM 2.0 synthesis completed! Open **Knowledge Analysis** to explore the full report."
                    )
    except (AIServiceError, HistoryStorageError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected Multi-LLM pipeline error: {exc}")

st.divider()

# One-click pipeline runner for full hackathon workflow
action_col1, action_col2 = st.columns([2, 3])
with action_col1:
    run_all = st.button(
        "Run Full AI Synthesis Pipeline",
        type="primary",
        width="stretch",
        disabled=not configured,
        help="Runs Knowledge Analysis, Common Concept extraction, Research Gap Detection, Knowledge Map, and auto-saves to Research History.",
    )
with action_col2:
    if not configured:
        st.info("Add `GEMINI_API_KEY` in `.env` (or sidebar override) to run AI synthesis.")
    else:
        st.caption(
            f"Active model: `{st.session_state.selected_model}` · "
            "Runs structured analysis, cross-paper concepts, gap detection, map generation, and saves to Research History."
        )

if run_all:
    api_key = st.session_state.get("api_key_override")
    model = st.session_state.get("selected_model")
    allow_fallback = st.session_state.get("auto_fallback", True)
    fallback_notes: list[str] = []
    try:
        with st.status("Running KNOWLEDGE X AI pipeline...", expanded=True) as status:
            st.write("1/3 Extracting structured knowledge analysis and common concepts...")
            analysis = analyze_documents(
                docs, model=model, api_key=api_key, allow_fallback=allow_fallback
            )
            st.session_state.analysis_result = analysis
            if analysis.get("_fallback_note"):
                fallback_notes.append(analysis["_fallback_note"])

            st.write("2/3 Detecting cross-document research gaps...")
            gaps = detect_research_gaps(
                docs,
                analysis_result=analysis,
                model=model,
                api_key=api_key,
                allow_fallback=allow_fallback,
            )
            st.session_state.gap_report = gaps
            if gaps.get("_fallback_note"):
                fallback_notes.append(gaps["_fallback_note"])

            st.write("3/3 Building knowledge graph & saving to Research History...")
            graph_data = build_structural_graph_data(
                docs, analysis_result=analysis, gap_report=gaps
            )
            st.session_state.knowledge_graph_data = graph_data

            saved_id = save_research_session(
                documents=docs,
                analysis_results=analysis,
                research_gaps=gaps,
                knowledge_graph=graph_data,
                session_id=st.session_state.get("current_session_id"),
                user_id=_active_user_id(),
            )
            st.session_state.current_session_id = saved_id
            st.session_state.active_history_session = None
            status.update(label="Full AI synthesis completed & saved.", state="complete", expanded=False)
        for note in dict.fromkeys(fallback_notes):
            st.info(note)
        st.success(
            f"Analysis, common concepts, research gaps, and knowledge map are ready "
            f"(saved to Research History as `{saved_id}`)."
        )
    except (AIServiceError, HistoryStorageError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected pipeline error: {exc}")

st.divider()
st.subheader("Uploaded Document Library")

doc_filter = st.text_input(
    "Filter documents",
    placeholder="Filter by filename, title, or author...",
    key="dashboard_doc_filter",
    label_visibility="collapsed",
).strip().lower()

filtered_docs = [
    d for d in docs
    if not doc_filter
    or doc_filter in d.get("filename", "").lower()
    or doc_filter in d.get("title", "").lower()
    or doc_filter in d.get("author", "").lower()
]
if doc_filter and not filtered_docs:
    st.info(f"No uploaded documents match '{doc_filter}'.")

for doc in list(filtered_docs):
    with st.container():
        c_info, c_action = st.columns([5, 1])
        with c_info:
            source_badges = ""
            doc_chunks = doc.get("chunks") or []
            doc_indexed = sum(1 for ch in doc_chunks if ch.get("chunk_id") in indexed_emb_map)
            if doc_chunks:
                source_badges += f"<span class='kx-badge'>{len(doc_chunks)} chunks</span>"
                emb_cls = "kx-badge-ok" if doc_indexed == len(doc_chunks) else "kx-badge-medium"
                source_badges += f"<span class='kx-badge {emb_cls}'>{doc_indexed}/{len(doc_chunks)} embedded</span>"
            if doc.get("source_url"):
                source_badges += "<span class='kx-badge kx-badge-low'>Link Import</span>"
            if doc.get("is_scanned"):
                source_badges += "<span class='kx-badge kx-badge-medium'>OCR Scan</span>"
            st.markdown(
                f"<div class='kx-card'>"
                f"<div class='kx-card-title'>{escape(doc['filename'])}</div>"
                f"<div class='kx-card-body'>"
                f"<b>Title:</b> {escape(doc.get('title', doc['filename']))} &nbsp;&middot;&nbsp; "
                f"<b>Author:</b> {escape(doc.get('author', 'Unknown'))}"
                f"</div>"
                f"<div class='kx-badge-row'>"
                f"<span class='kx-badge kx-badge-info'>{doc['page_count']} pages</span>"
                f"<span class='kx-badge'>{doc['word_count']:,} words</span>"
                f"<span class='kx-badge'>{doc['file_size_kb']} KB</span>"
                f"{source_badges}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with c_action:
            st.write("")
            if st.button("Remove", key=f"remove_{doc['file_hash']}", width="stretch"):
                st.session_state.documents = remove_document(docs, doc["file_hash"], delete_from_disk=True)
                _reset_analysis_after_doc_change()
                st.session_state.uploader_nonce += 1
                st.rerun()

st.divider()
st.subheader("Extracted PDF Text Inspector")

selected_filename = st.selectbox(
    "Select document to inspect",
    options=[d["filename"] for d in docs],
)
selected_doc = next(d for d in docs if d["filename"] == selected_filename)

page_options = ["Full Extracted Text"] + [
    f"Page {p['page_number']} ({p['word_count']} words)"
    for p in selected_doc.get("pages", [])
]
selected_page_label = st.selectbox("Page view", options=page_options)

if selected_page_label == "Full Extracted Text":
    preview_text = selected_doc.get("full_text", "")
else:
    page_idx = page_options.index(selected_page_label) - 1
    preview_text = selected_doc["pages"][page_idx].get("text", "")

st.text_area(
    "Extracted text",
    value=preview_text,
    height=320,
    disabled=True,
    label_visibility="collapsed",
)
