"""PDF ingestion, validation, persistence, and text-context utilities."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlsplit

import pymupdf
import requests

from modules.history_db import (
    DEFAULT_DB_PATH,
    HistoryStorageError,
    load_cached_document_record,
    save_document_chunks,
)

DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 200
_COMMON_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "methodology",
    "methods",
    "materials and methods",
    "system architecture",
    "architecture",
    "experimental setup",
    "experiments",
    "results",
    "evaluation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
}
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,2}|[IVXLCDM]+|[A-Z])[\.\)]\s+[A-Za-z][^\n]{2,75}$"
)

MAX_URL_PDF_BYTES = 25 * 1024 * 1024  # 25 MB limit
MAX_URL_REDIRECTS = 5
_NAT64_WELL_KNOWN_NET = ipaddress.IPv6Network("64:ff9b::/96")
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
    "metadata",
}


class PDFProcessingError(Exception):
    """Raised when a PDF file cannot be validated or parsed."""


def sanitize_filename(filename: str) -> str:
    """Normalize an uploaded filename to a safe PDF basename."""
    base = Path(filename or "document.pdf").name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    if not cleaned:
        cleaned = "document.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _clean_page_text(raw_text: str) -> str:
    """Normalize whitespace while preserving logical paragraph breaks."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _detect_section_heading(line: str) -> str | None:
    """Detect whether a single line looks like a document section heading."""
    cleaned = re.sub(r"\s+", " ", (line or "").strip()).strip(":.- ")
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 80:
        return None
    if cleaned.endswith((".", "?", ";", ",")) and not _NUMBERED_HEADING_RE.match(line.strip()):
        return None
    lower = cleaned.lower()
    if lower in _COMMON_HEADINGS:
        return cleaned
    if _NUMBERED_HEADING_RE.match(line.strip()):
        return cleaned
    if cleaned.isupper() and any(ch.isalpha() for ch in cleaned) and len(cleaned.split()) <= 8:
        return cleaned.title()
    return None


def _split_oversized_paragraph(paragraph: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a single paragraph exceeding max_chars on sentence/word boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    step = max(200, max_chars - max(0, overlap_chars))
    start = 0
    while start < len(paragraph):
        end = min(len(paragraph), start + max_chars)
        if end < len(paragraph):
            split_at = max(paragraph.rfind(". ", start, end), paragraph.rfind(" ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        piece = paragraph[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(paragraph):
            break
        start = max(start + 1, end - overlap_chars)
    return pieces


def chunk_document_pages(
    pages: list[dict[str, Any]],
    filename: str,
    file_hash: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split all pages into ordered section-aware chunks without dropping middle content."""
    chunks: list[dict[str, Any]] = []
    current_section = "Document Start"
    chunk_index = 0

    for page in pages:
        pno = int(page.get("page_number") or 1)
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", page_text) if p.strip()]
        buffer = ""
        chunk_section = current_section

        for para in paragraphs:
            first_line = para.splitlines()[0] if para else ""
            detected = _detect_section_heading(first_line)
            if detected:
                if buffer.strip():
                    chunks.append(
                        {
                            "chunk_id": f"{file_hash}:{chunk_index}",
                            "file_hash": file_hash,
                            "filename": filename,
                            "page_number": pno,
                            "chunk_index": chunk_index,
                            "section_title": chunk_section,
                            "text": buffer.strip(),
                            "word_count": len(buffer.split()),
                            "char_count": len(buffer.strip()),
                        }
                    )
                    chunk_index += 1
                    buffer = ""
                current_section = detected
                chunk_section = detected

            for sub_para in _split_oversized_paragraph(para, chunk_chars, overlap_chars):
                candidate = f"{buffer}\n\n{sub_para}".strip() if buffer else sub_para
                if len(candidate) <= chunk_chars:
                    buffer = candidate
                else:
                    if buffer.strip():
                        chunks.append(
                            {
                                "chunk_id": f"{file_hash}:{chunk_index}",
                                "file_hash": file_hash,
                                "filename": filename,
                                "page_number": pno,
                                "chunk_index": chunk_index,
                                "section_title": chunk_section,
                                "text": buffer.strip(),
                                "word_count": len(buffer.split()),
                                "char_count": len(buffer.strip()),
                            }
                        )
                        chunk_index += 1
                    tail = buffer[-overlap_chars:].strip() if (overlap_chars > 0 and buffer) else ""
                    if tail and len(tail) + len(sub_para) + 2 <= chunk_chars:
                        buffer = f"{tail}\n{sub_para}".strip()
                    else:
                        buffer = sub_para
                    chunk_section = current_section

        if buffer.strip():
            chunks.append(
                {
                    "chunk_id": f"{file_hash}:{chunk_index}",
                    "file_hash": file_hash,
                    "filename": filename,
                    "page_number": pno,
                    "chunk_index": chunk_index,
                    "section_title": chunk_section,
                    "text": buffer.strip(),
                    "word_count": len(buffer.split()),
                    "char_count": len(buffer.strip()),
                }
            )
            chunk_index += 1

    return chunks


def _persist_pdf_bytes(
    file_bytes: bytes,
    safe_name: str,
    file_hash: str,
    save_dir: Path | str | None,
) -> tuple[str | None, str]:
    """Save PDF bytes to save_dir without overwriting distinct content with the same name."""
    if save_dir is None:
        return None, safe_name
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name
    if target_path.exists():
        existing_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
        if existing_hash != file_hash:
            target_path = target_dir / f"{target_path.stem}_{file_hash[:8]}.pdf"
            safe_name = target_path.name
    if not target_path.exists():
        target_path.write_bytes(file_bytes)
    return str(target_path), safe_name


def extract_pdf_bytes(
    file_bytes: bytes,
    filename: str,
    save_dir: Path | str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Validate and extract text, pages, section chunks, and metadata (reusing SQLite cache)."""
    safe_name = sanitize_filename(filename)
    if not file_bytes:
        raise PDFProcessingError(f"'{safe_name}' is empty (0 bytes).")
    if b"%PDF-" not in file_bytes[:1024]:
        raise PDFProcessingError(
            f"'{safe_name}' is not a valid PDF file (missing %PDF- header)."
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Reuse cached extracted text & chunks when the same SHA-256 PDF is seen again
    try:
        cached = load_cached_document_record(file_hash, filename_override=safe_name, db_path=db_path)
    except HistoryStorageError:
        cached = None

    if cached is not None:
        saved_path, final_name = _persist_pdf_bytes(file_bytes, safe_name, file_hash, save_dir)
        cached["filename"] = final_name
        cached["saved_path"] = saved_path
        for ch in cached.get("chunks", []):
            ch["filename"] = final_name
        return cached

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(
            f"'{safe_name}' is not a valid or readable PDF ({exc})."
        ) from exc

    try:
        if getattr(doc, "needs_pass", False):
            raise PDFProcessingError(
                f"'{safe_name}' is password-protected and cannot be read."
            )

        if doc.page_count == 0:
            raise PDFProcessingError(f"'{safe_name}' contains zero pages.")

        pages: list[dict[str, Any]] = []
        formatted_pages: list[str] = []

        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            cleaned = _clean_page_text(page.get_text("text") or "")
            words = cleaned.split()
            pages.append(
                {
                    "page_number": idx + 1,
                    "text": cleaned,
                    "word_count": len(words),
                    "char_count": len(cleaned),
                }
            )
            if cleaned:
                formatted_pages.append(f"[Page {idx + 1}]\n{cleaned}")

        full_text = "\n\n".join(formatted_pages).strip()
        if not full_text:
            raise PDFProcessingError(
                f"'{safe_name}' has no extractable text (it may be a scanned image-only PDF)."
            )

        raw_meta = doc.metadata or {}
        title = (raw_meta.get("title") or "").strip() or Path(safe_name).stem
        author = (raw_meta.get("author") or "").strip() or "Unknown"
        subject = (raw_meta.get("subject") or "").strip()
        page_count = doc.page_count
    finally:
        doc.close()

    saved_path, safe_name = _persist_pdf_bytes(file_bytes, safe_name, file_hash, save_dir)

    total_words = sum(p["word_count"] for p in pages)
    total_chars = sum(p["char_count"] for p in pages)
    chunks = chunk_document_pages(pages, filename=safe_name, file_hash=file_hash)

    record = {
        "filename": safe_name,
        "title": title,
        "author": author,
        "subject": subject,
        "file_hash": file_hash,
        "file_size_kb": round(len(file_bytes) / 1024.0, 1),
        "page_count": page_count,
        "word_count": total_words,
        "char_count": total_chars,
        "pages": pages,
        "full_text": full_text,
        "saved_path": saved_path,
        "chunks": chunks,
        "from_cache": False,
    }

    try:
        save_document_chunks(record, chunks, db_path=db_path)
    except HistoryStorageError:
        pass

    return record


def process_uploaded_files(
    uploaded_files: Iterable[Any],
    existing_docs: list[dict[str, Any]],
    save_dir: Path | str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Ingest multiple uploaded files without duplicating already-loaded PDFs."""
    updated = list(existing_docs)
    known_hashes = {doc["file_hash"] for doc in updated}
    added_names: list[str] = []
    errors: list[str] = []

    for uploaded in uploaded_files or []:
        name = getattr(uploaded, "name", "uploaded.pdf")
        try:
            raw_bytes = uploaded.getvalue()
            quick_hash = hashlib.sha256(raw_bytes).hexdigest()
            if quick_hash in known_hashes:
                continue
            record = extract_pdf_bytes(raw_bytes, name, save_dir=save_dir, db_path=db_path)
            updated.append(record)
            known_hashes.add(record["file_hash"])
            added_names.append(record["filename"])
        except PDFProcessingError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Unexpected error processing '{name}': {exc}")

    return updated, added_names, errors


def load_existing_uploads(
    uploads_dir: Path | str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load any PDF files currently stored in the uploads directory (using SQLite cache)."""
    directory = Path(uploads_dir)
    if not directory.exists():
        return [], []

    docs: list[dict[str, Any]] = []
    known_hashes: set[str] = set()
    errors: list[str] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            raw_bytes = pdf_path.read_bytes()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()
            if file_hash in known_hashes:
                continue
            record = extract_pdf_bytes(raw_bytes, pdf_path.name, save_dir=None, db_path=db_path)
            record["saved_path"] = str(pdf_path)
            docs.append(record)
            known_hashes.add(file_hash)
        except PDFProcessingError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"Could not load '{pdf_path.name}': {exc}")

    return docs, errors


def remove_document(
    documents: list[dict[str, Any]],
    file_hash: str,
    delete_from_disk: bool = True,
) -> list[dict[str, Any]]:
    """Remove a single document by hash and optionally delete its saved upload."""
    remaining: list[dict[str, Any]] = []
    for doc in documents:
        if doc["file_hash"] == file_hash:
            if delete_from_disk and doc.get("saved_path"):
                path = Path(doc["saved_path"])
                if path.exists():
                    path.unlink(missing_ok=True)
            continue
        remaining.append(doc)
    return remaining


def clear_all_uploads(uploads_dir: Path | str) -> None:
    """Remove all saved PDF files from the uploads directory."""
    directory = Path(uploads_dir)
    if not directory.exists():
        return
    for pdf_path in directory.glob("*.pdf"):
        pdf_path.unlink(missing_ok=True)


def compute_corpus_stats(documents: list[dict[str, Any]]) -> dict[str, int]:
    """Compute aggregate statistics strictly from loaded documents."""
    return {
        "document_count": len(documents),
        "total_pages": sum(doc.get("page_count", 0) for doc in documents),
        "total_words": sum(doc.get("word_count", 0) for doc in documents),
        "total_chars": sum(doc.get("char_count", 0) for doc in documents),
    }


def build_corpus_context(documents: list[dict[str, Any]], max_chars: int = 120000) -> str:
    """Build a balanced multi-document context string within the character budget."""
    if not documents:
        return ""

    per_doc_budget = max(2500, max_chars // len(documents))
    sections: list[str] = []

    for doc in documents:
        header = (
            f"=== DOCUMENT: {doc['filename']} "
            f"(Title: {doc.get('title', doc['filename'])}, "
            f"Pages: {doc.get('page_count', 0)}, Words: {doc.get('word_count', 0)}) ==="
        )
        body = doc.get("full_text", "")
        if len(body) > per_doc_budget:
            head_len = int(per_doc_budget * 0.72)
            tail_len = per_doc_budget - head_len - 80
            body = (
                f"{body[:head_len].rstrip()}\n\n"
                f"[... middle excerpt truncated for length ({len(body)} total chars) ...]\n\n"
                f"{body[-max(0, tail_len):].lstrip()}"
            )
        sections.append(f"{header}\n{body}")

    return "\n\n".join(sections)


def _score_chunk_text(snippet: str, query_lower: str, query_tokens: list[str]) -> float:
    """Compute lexical relevance score for a single chunk snippet."""
    lower = snippet.lower()
    score = 0.0
    for tok in query_tokens:
        hits = lower.count(tok)
        if hits:
            score += 1.0 + min(hits, 5) * 0.35
    if query_lower and query_lower in lower:
        score += 3.0
    return score


def _lexical_retrieve_chunks(
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 10,
    chunk_chars: int = 1600,
) -> list[dict[str, Any]]:
    """Retrieve the highest-scoring document page chunks using lexical keyword matching."""
    if not documents:
        return []

    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "what", "which",
        "are", "was", "were", "how", "why", "who", "when", "where", "about",
        "into", "across", "between", "paper", "papers", "document", "documents",
        "pdf", "pdfs", "can", "you", "explain", "tell", "does", "have", "has",
    }
    query_lower = query.strip().lower()
    query_tokens = [
        tok for tok in re.findall(r"[a-z0-9]{2,}", query_lower)
        if tok not in stop_words
    ]

    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        doc_chunks: list[dict[str, Any]] = []
        cached_chunks = doc.get("chunks")
        if cached_chunks:
            for ch in cached_chunks:
                snippet = (ch.get("text") or "").strip()
                if not snippet:
                    continue
                doc_chunks.append(
                    {
                        "chunk_id": ch.get("chunk_id"),
                        "file_hash": ch.get("file_hash") or doc.get("file_hash"),
                        "filename": ch.get("filename") or doc["filename"],
                        "page_number": int(ch.get("page_number") or 1),
                        "chunk_index": int(ch.get("chunk_index") or 0),
                        "section_title": str(ch.get("section_title") or ""),
                        "text": snippet,
                        "score": _score_chunk_text(snippet, query_lower, query_tokens),
                    }
                )
        else:
            for page in doc.get("pages", []):
                text = page.get("text", "").strip()
                if not text:
                    continue
                step = max(800, chunk_chars - 200)
                for start in range(0, len(text), step):
                    snippet = text[start : start + chunk_chars].strip()
                    if not snippet:
                        continue
                    doc_chunks.append(
                        {
                            "filename": doc["filename"],
                            "page_number": page["page_number"],
                            "text": snippet,
                            "score": _score_chunk_text(snippet, query_lower, query_tokens),
                        }
                    )
        # Ensure at least the top chunk from each document is kept as candidate
        doc_chunks.sort(key=lambda c: c["score"], reverse=True)
        if doc_chunks:
            all_chunks.extend(doc_chunks[: max(2, top_k // max(1, len(documents)) + 1)])

    all_chunks.sort(key=lambda c: c["score"], reverse=True)
    if not all_chunks:
        return []

    # If no lexical terms matched (e.g. broad summary question), take opening chunks evenly
    if all_chunks[0]["score"] == 0:
        broad: list[dict[str, Any]] = []
        for doc in documents:
            first_pages = [p for p in doc.get("pages", []) if p.get("text")]
            for p in first_pages[:2]:
                broad.append(
                    {
                        "filename": doc["filename"],
                        "page_number": p["page_number"],
                        "text": p["text"][:chunk_chars],
                        "score": 0.1,
                    }
                )
        return broad[:top_k]

    return all_chunks[:top_k]


def retrieve_relevant_chunks(
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 10,
    chunk_chars: int = 1600,
    use_semantic: bool = True,
    embedding_model: str | None = None,
    api_key: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    ranking_config: Any = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval (semantic Gemini embeddings + lexical RRF) with automatic lexical fallback."""
    lexical_results = _lexical_retrieve_chunks(
        documents=documents,
        query=query,
        top_k=max(top_k, 10),
        chunk_chars=chunk_chars,
    )
    if not use_semantic or not documents or not (query or "").strip():
        return lexical_results[:top_k]

    try:
        from modules.ai_analysis import DEFAULT_EMBEDDING_MODEL
        from modules.semantic_search import reciprocal_rank_fusion, semantic_search_chunks

        semantic_results = semantic_search_chunks(
            query=query,
            documents=documents,
            top_k=max(top_k, 10),
            model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            api_key=api_key,
            db_path=db_path,
            auto_index=True,
        )
        if not semantic_results:
            return lexical_results[:top_k]
        return reciprocal_rank_fusion(
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            top_k=top_k,
            config=ranking_config,
        )
    except Exception:
        return lexical_results[:top_k]


def _is_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only when an IP address is publicly routable."""
    if isinstance(ip_obj, ipaddress.IPv6Address):
        if ip_obj.ipv4_mapped is not None:
            ip_obj = ip_obj.ipv4_mapped
        elif ip_obj in _NAT64_WELL_KNOWN_NET:
            ip_obj = ipaddress.IPv4Address(int(ip_obj) & 0xFFFFFFFF)
    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        return False
    return ip_obj.is_global


def validate_public_pdf_url(url: str) -> str:
    """Validate that a URL uses HTTP(S) and resolves strictly to public IP addresses."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise PDFProcessingError("Please enter a PDF URL.")

    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise PDFProcessingError(
            "Invalid URL scheme. Only public http:// and https:// URLs are supported."
        )

    if parsed.username or parsed.password:
        raise PDFProcessingError("URLs containing embedded credentials are not allowed.")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise PDFProcessingError("Invalid URL: missing hostname.")

    if (
        hostname in _BLOCKED_HOSTNAMES
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
        or hostname.endswith(".arpa")
    ):
        raise PDFProcessingError(
            f"Access to local or internal address '{hostname}' is blocked for security."
        )

    # Check literal IP addresses first
    try:
        literal_ip = ipaddress.ip_address(hostname)
        if not _is_public_ip(literal_ip):
            raise PDFProcessingError(
                f"Access to private or loopback IP '{hostname}' is blocked for security."
            )
    except ValueError:
        pass

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise PDFProcessingError(
            f"Could not resolve hostname '{hostname}' ({exc})."
        ) from exc

    resolved_ips: list[str] = []
    for entry in addr_info:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0].split("%")[0]
        resolved_ips.append(ip_str)
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not _is_public_ip(ip_obj):
            raise PDFProcessingError(
                f"URL target '{hostname}' resolves to unsafe non-public IP '{ip_str}'."
            )

    if not resolved_ips:
        raise PDFProcessingError(f"Hostname '{hostname}' did not resolve to any valid IP address.")

    return cleaned


def _filename_from_response(url: str, headers: Any) -> str:
    """Determine a safe .pdf filename from Content-Disposition or URL path."""
    content_disp = headers.get("Content-Disposition", "") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";\r\n]+)"?', content_disp, re.IGNORECASE)
    if match:
        candidate = unquote(match.group(1).strip())
        if candidate:
            return sanitize_filename(candidate)

    parsed_path = unquote(urlsplit(url).path or "")
    tail = Path(parsed_path).name
    if tail:
        return sanitize_filename(tail)
    return "linked_document.pdf"


def fetch_pdf_from_url(
    url: str,
    existing_docs: list[dict[str, Any]],
    save_dir: Path | str,
    max_bytes: int = MAX_URL_PDF_BYTES,
    timeout_seconds: int = 25,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """Securely download a public PDF URL, validate type/size, and extract text via PyMuPDF."""
    current_url = validate_public_pdf_url(url)
    headers = {
        "User-Agent": "KNOWLEDGE-X-Research-Assistant/1.0 (+https://streamlit.io)",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    response: requests.Response | None = None
    try:
        for hop in range(MAX_URL_REDIRECTS + 1):
            validate_public_pdf_url(current_url)
            response = requests.get(
                current_url,
                headers=headers,
                stream=True,
                timeout=(10, timeout_seconds),
                allow_redirects=False,
            )
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.close()
                response = None
                if not location:
                    raise PDFProcessingError("Redirect response was missing a Location header.")
                if hop >= MAX_URL_REDIRECTS:
                    raise PDFProcessingError("Too many redirects while downloading PDF.")
                current_url = urljoin(current_url, location)
                continue
            break

        assert response is not None
        if response.status_code != 200:
            raise PDFProcessingError(
                f"Download failed with HTTP {response.status_code} ({response.reason or 'Error'})."
            )

        content_length_raw = response.headers.get("Content-Length")
        if content_length_raw:
            try:
                content_length = int(content_length_raw)
                if content_length > max_bytes:
                    raise PDFProcessingError(
                        f"Remote PDF exceeds maximum allowed size ({max_bytes // (1024 * 1024)} MB)."
                    )
            except ValueError:
                pass

        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise PDFProcessingError(
                    f"Downloaded file exceeds maximum allowed size ({max_bytes // (1024 * 1024)} MB)."
                )
            chunks.append(chunk)

        raw_bytes = b"".join(chunks)
        if not raw_bytes:
            raise PDFProcessingError("Remote URL returned an empty response (0 bytes).")

        if b"%PDF-" not in raw_bytes[:1024]:
            type_hint = f" (Content-Type: {content_type})" if content_type else ""
            raise PDFProcessingError(
                f"The URL did not return a valid PDF document{type_hint}. "
                "Please provide a direct link to a PDF file."
            )

        filename = _filename_from_response(current_url, response.headers)
    except PDFProcessingError:
        raise
    except requests.exceptions.Timeout as exc:
        raise PDFProcessingError("Download timed out while fetching the PDF URL.") from exc
    except requests.exceptions.SSLError as exc:
        raise PDFProcessingError(f"SSL verification failed for URL: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise PDFProcessingError(f"Could not download PDF from URL: {exc}") from exc
    finally:
        if response is not None:
            response.close()

    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    for doc in existing_docs:
        if doc["file_hash"] == file_hash:
            return (
                list(existing_docs),
                doc,
                f"'{doc['filename']}' is already loaded in the workspace.",
            )

    record = extract_pdf_bytes(raw_bytes, filename, save_dir=save_dir, db_path=db_path)
    record["source_url"] = current_url
    updated = list(existing_docs) + [record]
    return (
        updated,
        record,
        f"Downloaded and processed '{record['filename']}' ({record['page_count']} pages, {record['word_count']:,} words).",
    )

