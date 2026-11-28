"""Phase 1 unit tests: section-aware PDF chunking, SQLite document_chunks persistence, and cache reuse."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from modules.history_db import (
    get_document_chunks_by_hash,
    has_cached_document,
    init_history_db,
    load_cached_document_record,
)
from modules.pdf_reader import (
    build_corpus_context,
    compute_corpus_stats,
    extract_pdf_bytes,
    load_existing_uploads,
    process_uploaded_files,
    retrieve_relevant_chunks,
)


def _make_pdf_bytes(pages_text: list[str], title: str = "Test Paper") -> bytes:
    """Create a valid multi-page PDF in memory using PyMuPDF."""
    doc = pymupdf.open()
    doc.set_metadata({"title": title, "author": "Test Author"})
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class TestPhase1PdfProcessingAndStorage(unittest.TestCase):
    def test_section_aware_chunking_preserves_middle_pages_and_metadata(self) -> None:
        pages_text = [
            "1. Introduction\n\nThis paper studies multi-document synthesis across research corpora.",
            "2. Methodology\n\nMiddle section detail: hybrid chunking keeps all pages intact without truncation.",
            "3. Results\n\nEvaluation demonstrates 94% recall on cross-paper retrieval benchmarks.",
        ]
        pdf_bytes = _make_pdf_bytes(pages_text, title="Sectioned Paper")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "test.db"
            uploads = tmp / "uploads"

            record = extract_pdf_bytes(pdf_bytes, "sectioned.pdf", save_dir=uploads, db_path=db_path)
            self.assertFalse(record["from_cache"])
            self.assertEqual(record["page_count"], 3)
            self.assertGreaterEqual(len(record["chunks"]), 3)

            sections = [c["section_title"] for c in record["chunks"]]
            self.assertIn("1. Introduction", sections)
            self.assertIn("2. Methodology", sections)
            self.assertIn("3. Results", sections)

            # Verify middle page content is stored and indexed sequentially
            indices = [c["chunk_index"] for c in record["chunks"]]
            self.assertEqual(indices, list(range(len(record["chunks"]))))
            self.assertTrue(any("Middle section detail" in c["text"] for c in record["chunks"]))
            self.assertTrue(all(c["filename"] == "sectioned.pdf" for c in record["chunks"]))
            self.assertTrue(all(c["file_hash"] == record["file_hash"] for c in record["chunks"]))

            # Verify SQLite persistence and indexes
            self.assertTrue(has_cached_document(record["file_hash"], db_path=db_path))
            db_chunks = get_document_chunks_by_hash(record["file_hash"], db_path=db_path)
            self.assertEqual(len(db_chunks), len(record["chunks"]))

            conn = sqlite3.connect(str(db_path))
            indexes = {
                row[1]
                for row in conn.execute("PRAGMA index_list(document_chunks)").fetchall()
            }
            conn.close()
            self.assertIn("idx_document_chunks_file_hash", indexes)
            self.assertIn("idx_document_chunks_page", indexes)
            self.assertIn("idx_document_chunks_filename", indexes)

    def test_reupload_reuses_sqlite_cache_without_reparsing_pdf(self) -> None:
        pdf_bytes = _make_pdf_bytes(
            [
                "Abstract\n\nCached extraction test abstract.",
                "Conclusion\n\nFinal cached conclusion paragraph.",
            ],
            title="Cached Paper",
        )

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "test_cache.db"
            uploads = tmp / "uploads"

            first = extract_pdf_bytes(pdf_bytes, "paper_a.pdf", save_dir=uploads, db_path=db_path)
            self.assertFalse(first["from_cache"])

            # Re-uploading same bytes under a new filename must hit SQLite cache and skip pymupdf.open
            with patch("modules.pdf_reader.pymupdf.open", side_effect=AssertionError("Should use SQLite cache!")):
                second = extract_pdf_bytes(pdf_bytes, "paper_copy.pdf", save_dir=uploads, db_path=db_path)
                self.assertTrue(second["from_cache"])
                self.assertEqual(second["file_hash"], first["file_hash"])
                self.assertEqual(second["filename"], "paper_copy.pdf")
                self.assertEqual(len(second["chunks"]), len(first["chunks"]))
                self.assertEqual(second["page_count"], 2)

                # load_existing_uploads should also reuse cache
                loaded_docs, errs = load_existing_uploads(uploads, db_path=db_path)
                self.assertEqual(errs, [])
                self.assertEqual(len(loaded_docs), 1)
                self.assertTrue(loaded_docs[0]["from_cache"])

    def test_duplicate_upload_deduplication_and_chat_compatibility(self) -> None:
        pdf1 = _make_pdf_bytes(["Introduction\n\nAlpha transformer scaling study."], title="Alpha")
        pdf2 = _make_pdf_bytes(["Methods\n\nBeta retrieval-augmented generation pipeline."], title="Beta")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "test_dup.db"
            uploads = tmp / "uploads"

            docs, added, errs = process_uploaded_files(
                [_FakeUpload("alpha.pdf", pdf1), _FakeUpload("alpha_dup.pdf", pdf1), _FakeUpload("beta.pdf", pdf2)],
                existing_docs=[],
                save_dir=uploads,
                db_path=db_path,
            )
            self.assertEqual(errs, [])
            self.assertEqual(len(docs), 2)
            self.assertEqual(added, ["alpha.pdf", "beta.pdf"])

            stats = compute_corpus_stats(docs)
            self.assertEqual(stats["document_count"], 2)
            self.assertEqual(stats["total_pages"], 2)

            ctx = build_corpus_context(docs)
            self.assertIn("alpha.pdf", ctx)
            self.assertIn("beta.pdf", ctx)

            hits = retrieve_relevant_chunks(docs, "retrieval augmented generation", db_path=db_path)
            self.assertTrue(len(hits) >= 1)
            self.assertEqual(hits[0]["filename"], "beta.pdf")
            self.assertEqual(hits[0]["page_number"], 1)
            self.assertEqual(hits[0]["section_title"], "Methods")

    def test_large_document_chunks_never_truncate_middle(self) -> None:
        pages = [
            f"Section {i}\n\n" + (f"Page {i} unique_marker_{i} research content. " * 40)
            for i in range(1, 16)
        ]
        pdf_bytes = _make_pdf_bytes(pages, title="Large 15-Page Paper")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            db_path = tmp / "test_large.db"
            record = extract_pdf_bytes(pdf_bytes, "large15.pdf", save_dir=None, db_path=db_path)

            all_text = "\n".join(c["text"] for c in record["chunks"])
            for i in range(1, 16):
                self.assertIn(f"unique_marker_{i}", all_text)

            restored = load_cached_document_record(record["file_hash"], db_path=db_path)
            self.assertIsNotNone(restored)
            restored_text = "\n".join(c["text"] for c in restored["chunks"])
            self.assertIn("unique_marker_8", restored_text)

    def test_existing_tables_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "preserve.db"
            init_history_db(db_path)
            conn = sqlite3.connect(str(db_path))
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            conn.close()
            self.assertIn("research_sessions", tables)
            self.assertIn("ambiguity_history", tables)
            self.assertIn("document_chunks", tables)


if __name__ == "__main__":
    unittest.main()
