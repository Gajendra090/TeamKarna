"""Phase 4 integration tests: Research Assistant hybrid search, Dashboard/Analysis Multi-LLM wiring, and fallback."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.ai_analysis import AIServiceError, answer_research_question
from modules.multi_llm_pipeline import run_multi_llm_pipeline
from modules.semantic_search import index_document_embeddings


def _doc(name: str, fhash: str, pno: int, sec: str, text: str) -> dict:
    return {
        "filename": name,
        "title": name.replace(".pdf", ""),
        "file_hash": fhash,
        "page_count": 1,
        "pages": [{"page_number": pno, "text": text, "word_count": len(text.split()), "char_count": len(text)}],
        "chunks": [
            {
                "chunk_id": f"{fhash}:0",
                "file_hash": fhash,
                "filename": name,
                "page_number": pno,
                "chunk_index": 0,
                "section_title": sec,
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
            }
        ],
    }


class TestPhase4Integration(unittest.TestCase):
    def test_research_assistant_reports_hybrid_mode_and_preserves_citations(self) -> None:
        docs = [
            _doc("paper_a.pdf", "h_a", 2, "Methods", "Transformer attention improves multi-PDF recall."),
            _doc("paper_b.pdf", "h_b", 5, "Results", "Hybrid retrieval boosts clinical accuracy to 95%."),
        ]

        def fake_embed(texts, model="gemini-embedding-001", **kwargs):
            return [[0.9, 0.2] if ("hybrid" in t.lower() or "accuracy" in t.lower()) else [0.1, 0.9] for t in texts], model

        def fake_json(*, prompt, schema, system_instruction, model, **kwargs):
            return {
                "answer": "Hybrid retrieval boosts clinical accuracy to 95%.",
                "citations": [{"filename": "paper_b.pdf", "page_number": 5, "evidence": "95% accuracy"}],
                "confidence": "High",
                "follow_up_questions": ["What dataset was evaluated?"],
                "_model_used": model,
            }

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ra_hybrid.db"
            with patch("modules.semantic_search.generate_embeddings", side_effect=fake_embed), patch(
                "modules.ai_analysis.generate_structured_json", side_effect=fake_json
            ):
                res = answer_research_question(
                    question="How much does hybrid retrieval improve accuracy?",
                    documents=docs,
                    use_semantic=True,
                    db_path=db_path,
                )
                self.assertEqual(res["_retrieval_mode"], "hybrid")
                self.assertEqual(res["citations"][0]["filename"], "paper_b.pdf")
                self.assertEqual(res["citations"][0]["page_number"], 5)

    def test_research_assistant_falls_back_to_lexical_when_embeddings_fail(self) -> None:
        docs = [_doc("paper_a.pdf", "h_a", 3, "Evaluation", "Latency benchmark measured at 120ms.")]

        def fake_json(*, prompt, schema, system_instruction, model, **kwargs):
            self.assertIn("paper_a.pdf", prompt)
            self.assertIn("Page: 3", prompt)
            return {
                "answer": "Measured latency is 120ms.",
                "citations": [{"filename": "paper_a.pdf", "page_number": 3, "evidence": "120ms"}],
                "confidence": "High",
                "follow_up_questions": [],
                "_model_used": model,
            }

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ra_fallback.db"
            with patch("modules.semantic_search.semantic_search_chunks", side_effect=AIServiceError("429 Quota")), patch(
                "modules.ai_analysis.generate_structured_json", side_effect=fake_json
            ):
                res = answer_research_question(
                    question="What was the latency benchmark?",
                    documents=docs,
                    use_semantic=True,
                    db_path=db_path,
                )
                self.assertEqual(res["_retrieval_mode"], "lexical")
                self.assertEqual(res["citations"][0]["page_number"], 3)

    def test_dashboard_indexing_and_multi_llm_cache_reuse(self) -> None:
        docs = [_doc("paper_a.pdf", "h_a", 1, "Intro", "Research synthesis study.")]
        embed_calls = 0

        def fake_embed(texts, model="gemini-embedding-001", **kwargs):
            nonlocal embed_calls
            embed_calls += 1
            return [[0.5, 0.5] for _ in texts], model

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "dash.db"
            with patch("modules.semantic_search.generate_embeddings", side_effect=fake_embed):
                s1 = index_document_embeddings(docs, db_path=db_path)
                self.assertEqual(s1["indexed_count"], 1)
                self.assertEqual(embed_calls, 1)

                s2 = index_document_embeddings(docs, db_path=db_path)
                self.assertEqual(s2["indexed_count"], 0)
                self.assertEqual(s2["cached_count"], 1)
                self.assertEqual(embed_calls, 1)

            # Empty doc list raises clean AIServiceError without crashing
            with self.assertRaises(AIServiceError):
                run_multi_llm_pipeline([], db_path=db_path)


if __name__ == "__main__":
    unittest.main()
