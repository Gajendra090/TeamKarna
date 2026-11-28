"""Phase 5 quality tests: SQLite WAL safety, vectorized search equivalence, PDF validation, sanitization, and hygiene."""

from __future__ import annotations

import ast
import glob
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.ai_analysis import _sanitize_secret_text
from modules.auth import init_auth_db
from modules.history_db import init_history_db, save_chunk_embeddings
from modules.pdf_reader import PDFProcessingError, extract_pdf_bytes
from modules.semantic_search import cosine_similarity, semantic_search_chunks


class TestPhase5Quality(unittest.TestCase):
    def test_sqlite_wal_configured_once_and_preserves_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "wal_test.db"
            init_history_db(db_path)
            init_auth_db(db_path)

            conn = sqlite3.connect(str(db_path))
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            self.assertEqual(str(mode).lower(), "wal")

            # Subsequent init calls should not fail or lose data
            init_history_db(db_path)
            init_auth_db(db_path)

    def test_vectorized_semantic_search_matches_pairwise_cosine(self) -> None:
        docs = [
            {
                "filename": "a.pdf",
                "file_hash": "ha",
                "chunks": [
                    {"chunk_id": "ha:0", "file_hash": "ha", "filename": "a.pdf", "page_number": 1, "chunk_index": 0, "section_title": "Intro", "text": "Alpha"},
                    {"chunk_id": "ha:1", "file_hash": "ha", "filename": "a.pdf", "page_number": 2, "chunk_index": 1, "section_title": "Body", "text": "Beta"},
                ],
            }
        ]
        vecs = {"ha:0": [0.8, 0.2, 0.1], "ha:1": [0.1, 0.9, 0.2]}
        q_vec = [0.75, 0.25, 0.0]

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "vec.db"
            save_chunk_embeddings(
                [
                    {
                        "chunk_id": cid,
                        "file_hash": "ha",
                        "embedding_model": "gemini-embedding-001",
                        "embedding_dimensions": 3,
                        "embedding_vector": v,
                    }
                    for cid, v in vecs.items()
                ],
                db_path=db_path,
            )

            with patch("modules.semantic_search.generate_embeddings", return_value=([q_vec], "gemini-embedding-001")):
                hits = semantic_search_chunks(
                    "alpha query",
                    docs,
                    top_k=2,
                    model="gemini-embedding-001",
                    db_path=db_path,
                    auto_index=False,
                )
            self.assertEqual(len(hits), 2)
            self.assertEqual(hits[0]["chunk_id"], "ha:0")
            self.assertAlmostEqual(hits[0]["semantic_score"], cosine_similarity(q_vec, vecs["ha:0"]), places=5)
            self.assertAlmostEqual(hits[1]["semantic_score"], cosine_similarity(q_vec, vecs["ha:1"]), places=5)

    def test_rejects_non_pdf_magic_bytes_and_empty_bytes(self) -> None:
        with self.assertRaises(PDFProcessingError):
            extract_pdf_bytes(b"", "empty.pdf")
        with self.assertRaises(PDFProcessingError) as ctx:
            extract_pdf_bytes(b"<html>not a pdf</html>", "fake.pdf")
        self.assertIn("%PDF-", str(ctx.exception))

    def test_secret_sanitization_preserves_diagnostics(self) -> None:
        raw = "HTTP 429 RESOURCE_EXHAUSTED on gemini-3.6-flash using key AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234"
        cleaned = _sanitize_secret_text(raw, "my-session-secret")
        self.assertNotIn("AIzaSy", cleaned)
        self.assertIn("[REDACTED_KEY]", cleaned)
        self.assertIn("HTTP 429 RESOURCE_EXHAUSTED", cleaned)
        self.assertIn("gemini-3.6-flash", cleaned)

    def test_no_unused_imports_in_project_python_files(self) -> None:
        py_files = sorted(
            glob.glob("*.py") + glob.glob("modules/*.py") + glob.glob("pages/*.py") + glob.glob("tests/*.py")
        )
        for path in py_files:
            src = Path(path).read_text(encoding="utf-8")
            tree = ast.parse(src, path)
            imported: dict[str, int] = {}
            used: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported[a.asname or a.name.split(".")[0]] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "__future__":
                        continue
                    for a in node.names:
                        imported[a.asname or a.name] = node.lineno
                elif isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    used.add(node.value.id)
            unused = [(k, v) for k, v in imported.items() if k not in used]
            self.assertEqual(unused, [], f"Unused imports in {path}: {unused}")


if __name__ == "__main__":
    unittest.main()
