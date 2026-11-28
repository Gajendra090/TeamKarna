"""Phase 3 unit tests: Gemini embeddings, SQLite chunk_embeddings, semantic search, and hybrid RRF."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.ai_analysis import AIServiceError, clear_daily_exhausted_models
from modules.history_db import get_chunk_embeddings, init_history_db, save_chunk_embeddings
from modules.pdf_reader import retrieve_relevant_chunks
from modules.semantic_search import (
    HybridRankingConfig,
    index_document_embeddings,
    reciprocal_rank_fusion,
    semantic_search_chunks,
)


def _doc(name: str, fhash: str, texts: list[tuple[int, str, str]]) -> dict:
    return {
        "filename": name,
        "file_hash": fhash,
        "page_count": len(texts),
        "chunks": [
            {
                "chunk_id": f"{fhash}:{idx}",
                "file_hash": fhash,
                "filename": name,
                "page_number": pno,
                "chunk_index": idx,
                "section_title": sec,
                "text": txt,
                "word_count": len(txt.split()),
                "char_count": len(txt),
            }
            for idx, (pno, sec, txt) in enumerate(texts)
        ],
    }


def _fake_vec(text: str) -> list[float]:
    t = text.lower()
    return [
        1.0 if ("quantum" in t or "qubit" in t or "entanglement" in t) else 0.1,
        1.0 if ("transformer" in t or "attention" in t or "neural" in t) else 0.1,
        1.0 if ("clinical" in t or "patient" in t or "diagnosis" in t) else 0.1,
    ]


class TestPhase3SemanticSearch(unittest.TestCase):
    def setUp(self) -> None:
        clear_daily_exhausted_models()

    def test_batch_embedding_generation_persistence_and_duplicate_prevention(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.models = self
                self.calls = 0

            def embed_content(self, *, model, contents, config):
                self.calls += 1
                return SimpleNamespace(
                    embeddings=[SimpleNamespace(values=_fake_vec(txt)) for txt in contents]
                )

        fake_client = FakeClient()
        docs = [
            _doc(
                "quantum.pdf",
                "h_q",
                [
                    (1, "Introduction", "Quantum entanglement enables fault-tolerant qubit gates."),
                    (2, "Methods", "Superconducting qubit calibration results."),
                ],
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "emb.db"
            with patch("modules.ai_analysis.resolve_api_key", return_value="fake-key"), patch(
                "modules.ai_analysis.get_gemini_client", return_value=fake_client
            ):
                first = index_document_embeddings(docs, model="gemini-embedding-001", db_path=db_path)
                self.assertEqual(first["indexed_count"], 2)
                self.assertEqual(first["cached_count"], 0)
                self.assertEqual(fake_client.calls, 1)  # Batched into 1 API call

                # Second run must reuse SQLite cache with 0 new API calls
                second = index_document_embeddings(docs, model="gemini-embedding-001", db_path=db_path)
                self.assertEqual(second["indexed_count"], 0)
                self.assertEqual(second["cached_count"], 2)
                self.assertEqual(fake_client.calls, 1)

                # Duplicate save_chunk_embeddings must insert 0 rows
                dup_saved = save_chunk_embeddings(
                    [
                        {
                            "chunk_id": "h_q:0",
                            "file_hash": "h_q",
                            "embedding_model": "gemini-embedding-001",
                            "embedding_dimensions": 3,
                            "embedding_vector": [1.0, 0.1, 0.1],
                        }
                    ],
                    db_path=db_path,
                )
                self.assertEqual(dup_saved, 0)

                stored = get_chunk_embeddings(["h_q"], "gemini-embedding-001", db_path=db_path)
                self.assertEqual(len(stored), 2)
                self.assertEqual(stored["h_q:0"]["embedding_dimensions"], 3)

    def test_semantic_similarity_ranking_and_model_reindexing(self) -> None:
        docs = [
            _doc("quantum.pdf", "h_q", [(3, "Results", "Quantum entanglement and qubit error correction.")]),
            _doc("nlp.pdf", "h_n", [(2, "Architecture", "Transformer self-attention neural networks.")]),
        ]

        def fake_embed(texts, model="gemini-embedding-001", **kwargs):
            return [_fake_vec(t) for t in texts], model

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "sem.db"
            with patch("modules.semantic_search.generate_embeddings", side_effect=fake_embed):
                hits = semantic_search_chunks(
                    "How do qubit gates reduce errors?",
                    docs,
                    top_k=2,
                    model="gemini-embedding-001",
                    db_path=db_path,
                )
                self.assertEqual(len(hits), 2)
                self.assertEqual(hits[0]["filename"], "quantum.pdf")
                self.assertEqual(hits[0]["page_number"], 3)
                self.assertEqual(hits[0]["section_title"], "Results")

                # Switching embedding model triggers clean re-indexing without deleting old model rows
                reindex = index_document_embeddings(docs, model="gemini-embedding-2", db_path=db_path)
                self.assertEqual(reindex["indexed_count"], 2)
                self.assertEqual(len(get_chunk_embeddings(["h_q", "h_n"], "gemini-embedding-001", db_path=db_path)), 2)
                self.assertEqual(len(get_chunk_embeddings(["h_q", "h_n"], "gemini-embedding-2", db_path=db_path)), 2)

    def test_hybrid_rrf_deduplication_and_lexical_fallback_on_api_failure(self) -> None:
        docs = [
            _doc("clinical.pdf", "h_c", [(4, "Evaluation", "Clinical patient diagnosis workflow accuracy.")]),
            _doc("nlp.pdf", "h_n", [(1, "Intro", "Transformer attention mechanisms.")]),
        ]

        lex = [
            {"chunk_id": "h_c:0", "filename": "clinical.pdf", "page_number": 4, "chunk_index": 0, "text": "Clinical", "score": 2.0},
            {"chunk_id": "h_n:0", "filename": "nlp.pdf", "page_number": 1, "chunk_index": 0, "text": "NLP", "score": 0.5},
        ]
        sem = [
            {"chunk_id": "h_c:0", "filename": "clinical.pdf", "page_number": 4, "chunk_index": 0, "text": "Clinical", "semantic_score": 0.98},
        ]
        fused = reciprocal_rank_fusion(
            lex,
            sem,
            top_k=5,
            config=HybridRankingConfig(rrf_k=60, lexical_weight=1.0, semantic_weight=1.5),
        )
        # Duplicate chunk h_c:0 is merged cleanly (2 unique chunks, not 3)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["chunk_id"], "h_c:0")
        self.assertEqual(fused[0]["retrieval_mode"], "hybrid")

        # When embedding API fails, retrieve_relevant_chunks falls back gracefully to lexical retrieval
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "fallback.db"
            with patch("modules.semantic_search.semantic_search_chunks", side_effect=AIServiceError("Quota exceeded")):
                res = retrieve_relevant_chunks(
                    docs,
                    "clinical patient diagnosis",
                    top_k=2,
                    use_semantic=True,
                    db_path=db_path,
                )
                self.assertTrue(len(res) >= 1)
                self.assertEqual(res[0]["filename"], "clinical.pdf")
                self.assertEqual(res[0]["page_number"], 4)

    def test_existing_tables_and_history_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "schema.db"
            init_history_db(db_path)
            conn = sqlite3.connect(str(db_path))
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            indexes = {r[1] for r in conn.execute("PRAGMA index_list(chunk_embeddings)").fetchall()}
            conn.close()
            self.assertIn("research_sessions", tables)
            self.assertIn("document_chunks", tables)
            self.assertIn("multi_llm_cache", tables)
            self.assertIn("chunk_embeddings", tables)
            self.assertIn("idx_chunk_embeddings_file_model", indexes)
            self.assertIn("idx_chunk_embeddings_chunk_id", indexes)


if __name__ == "__main__":
    unittest.main()
