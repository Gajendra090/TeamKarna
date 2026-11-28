"""Phase 2 unit tests: Multi-LLM parallel execution, caching/recovery, fallback, thread safety, and citations."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import errors as genai_errors

from modules.ai_analysis import (
    DEFAULT_MODEL,
    _get_daily_exhausted_snapshot,
    clear_daily_exhausted_models,
    generate_structured_json,
)
from modules.history_db import load_worker_cache, save_worker_cache
from modules.multi_llm_pipeline import (
    PROCESSING_VERSION,
    Worker1DocumentAnalysis,
    Worker2SectionBatch,
    Worker3ResearchIntelligence,
    MasterResearchReport,
    WorkerModelConfig,
    run_multi_llm_pipeline,
)


def _sample_doc(name: str, fhash: str, page_count: int = 2) -> dict:
    chunks = [
        {
            "chunk_id": f"{fhash}:{i}",
            "file_hash": fhash,
            "filename": name,
            "page_number": i + 1,
            "chunk_index": i,
            "section_title": f"Section {i + 1}",
            "text": f"{name} page {i + 1} empirical findings on neural retrieval.",
            "word_count": 9,
            "char_count": 64,
        }
        for i in range(page_count)
    ]
    return {
        "filename": name,
        "title": name.replace(".pdf", ""),
        "file_hash": fhash,
        "page_count": page_count,
        "chunks": chunks,
    }


def _fake_ai_response(schema, prompt: str, model: str) -> dict:
    if schema is Worker1DocumentAnalysis:
        fname = "paper_a.pdf" if "paper_a.pdf" in prompt else "paper_b.pdf"
        return {
            "filename": fname,
            "research_objectives": [f"Evaluate {fname} objective"],
            "core_thesis": f"Thesis of {fname}",
            "methodology": "Experimental benchmark",
            "key_findings": [{"finding": f"Finding from {fname}", "page_numbers": [1, 2]}],
            "limitations": ["Small sample size"],
            "key_terms": ["neural retrieval"],
            "_model_used": model,
            "_usage": {"prompt_tokens": 100, "output_tokens": 40, "total_tokens": 140},
        }
    if schema is Worker2SectionBatch:
        return {
            "section_summaries": [
                {
                    "section_title": "Section 1",
                    "page_numbers": [1],
                    "summary": "Introduces the task.",
                    "key_points": ["Point A"],
                },
                {
                    "section_title": "Section 2",
                    "page_numbers": [2],
                    "summary": "Reports benchmark metrics.",
                    "key_points": ["Point B"],
                },
            ],
            "_model_used": model,
            "_usage": {"prompt_tokens": 80, "output_tokens": 30, "total_tokens": 110},
        }
    if schema is Worker3ResearchIntelligence:
        return {
            "common_concepts": [
                {
                    "concept": "Neural Retrieval",
                    "explanation": "Shared across both papers.",
                    "documents": ["paper_a.pdf", "paper_b.pdf"],
                    "importance": "High",
                }
            ],
            "methodology_comparison": [
                {
                    "aspect": "Benchmark",
                    "comparison": "Both evaluate retrieval recall.",
                    "documents": ["paper_a.pdf", "paper_b.pdf"],
                }
            ],
            "similarities": ["Both target multi-PDF retrieval"],
            "contradictions": [],
            "research_gaps": [
                {
                    "title": "Latency under scale",
                    "description": "Missing 100+ PDF latency evaluation.",
                    "affected_documents": ["paper_a.pdf"],
                    "severity": "Medium",
                }
            ],
            "cross_document_insights": "Strong convergence on hybrid retrieval.",
            "_model_used": model,
            "_usage": {"prompt_tokens": 120, "output_tokens": 50, "total_tokens": 170},
        }
    if schema is MasterResearchReport:
        return {
            "executive_summary": "Combined synthesis of paper_a.pdf and paper_b.pdf.",
            "document_digests": [
                {"filename": "paper_a.pdf", "summary": "Summary A", "key_pages": [1, 2]},
                {"filename": "paper_b.pdf", "summary": "Summary B", "key_pages": [1, 2]},
            ],
            "synthesized_findings": [
                {
                    "claim": "Hybrid retrieval improves accuracy.",
                    "citations": [
                        {"filename": "paper_a.pdf", "page_number": 2},
                        {"filename": "paper_b.pdf", "page_number": 1},
                    ],
                }
            ],
            "common_concepts": [
                {
                    "concept": "Neural Retrieval",
                    "explanation": "Shared concept",
                    "documents": ["paper_a.pdf", "paper_b.pdf"],
                    "importance": "High",
                }
            ],
            "contradictions_and_gaps": ["Latency under scale"],
            "open_research_questions": ["How does chunk size affect recall?"],
            "source_references": [
                {"filename": "paper_a.pdf", "page_number": 2},
                {"filename": "paper_b.pdf", "page_number": 1},
            ],
            "_model_used": model,
            "_usage": {"prompt_tokens": 150, "output_tokens": 60, "total_tokens": 210},
        }
    raise AssertionError(f"Unexpected schema: {schema}")


class TestPhase2MultiLLMPipeline(unittest.TestCase):
    def setUp(self) -> None:
        clear_daily_exhausted_models()

    def test_parallel_execution_bounded_and_independent_models_and_citations(self) -> None:
        docs = [_sample_doc("paper_a.pdf", "hash_a"), _sample_doc("paper_b.pdf", "hash_b")]
        cfg = WorkerModelConfig(
            worker1_model="gemini-3.6-flash",
            worker2_model="gemini-3.5-flash-lite",
            worker3_model="gemini-3.8-flash",
            master_model="gemini-4-hydrogen",
        )

        active = 0
        peak = 0
        lock = threading.Lock()
        seen_models: list[str] = []

        def mocked_gen(*, prompt, schema, system_instruction, model, api_key=None, temperature=0.2, allow_fallback=True):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                seen_models.append(model)
            time.sleep(0.03)
            try:
                return _fake_ai_response(schema, prompt, model)
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "phase2.db"
            with patch("modules.multi_llm_pipeline.generate_structured_json", side_effect=mocked_gen):
                out = run_multi_llm_pipeline(
                    docs,
                    model_config=cfg,
                    max_workers=10,  # Should still be bounded to <= 3
                    db_path=db_path,
                )

            self.assertEqual(out["status"], "completed")
            self.assertLessEqual(peak, 3)
            self.assertGreaterEqual(peak, 2)
            self.assertEqual(len(out["worker1_results"]), 2)
            self.assertEqual(len(out["worker2_results"]), 2)
            self.assertIn("gemini-3.6-flash", seen_models)
            self.assertIn("gemini-3.5-flash-lite", seen_models)
            self.assertIn("gemini-3.8-flash", seen_models)
            self.assertIn("gemini-4-hydrogen", seen_models)

            refs = out["master_report"]["source_references"]
            self.assertEqual(
                {(r["filename"], r["page_number"]) for r in refs},
                {("paper_a.pdf", 2), ("paper_b.pdf", 1)},
            )
            self.assertGreater(out["token_usage"]["total_tokens"], 0)

    def test_caching_skips_completed_and_never_overwrites_valid_cache(self) -> None:
        docs = [_sample_doc("paper_a.pdf", "hash_a")]
        calls = 0

        def mocked_gen(*, prompt, schema, system_instruction, model, **kwargs):
            nonlocal calls
            calls += 1
            return _fake_ai_response(schema, prompt, model)

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "cache.db"
            with patch("modules.multi_llm_pipeline.generate_structured_json", side_effect=mocked_gen):
                first = run_multi_llm_pipeline(docs, db_path=db_path)
                self.assertEqual(first["status"], "completed")
                first_calls = calls
                self.assertEqual(first_calls, 4)  # W1 + W2 + W3 + Master

                # Second run must reuse SQLite cache with 0 Gemini API calls
                second = run_multi_llm_pipeline(docs, db_path=db_path)
                self.assertEqual(calls, first_calls)
                self.assertEqual(second["cache_hits"], 4)

            # Attempting to save a failed status must NOT overwrite completed cache
            save_worker_cache(
                file_hash="hash_a",
                worker_name="worker1_analysis",
                processing_version=PROCESSING_VERSION,
                model_used=DEFAULT_MODEL,
                status="failed",
                result_data=None,
                error_message="simulated transient failure",
                db_path=db_path,
            )
            cached = load_worker_cache("hash_a", "worker1_analysis", PROCESSING_VERSION, db_path=db_path)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["status"], "completed")

    def test_partial_worker_failure_does_not_abort_batch_and_retries(self) -> None:
        docs = [_sample_doc("paper_a.pdf", "hash_a"), _sample_doc("paper_b.pdf", "hash_b")]
        fail_b = True

        def mocked_gen(*, prompt, schema, system_instruction, model, **kwargs):
            if fail_b and schema is Worker1DocumentAnalysis and "paper_b.pdf" in prompt:
                raise RuntimeError("Simulated Worker 1 failure on paper_b.pdf")
            return _fake_ai_response(schema, prompt, model)

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "partial.db"
            with patch("modules.multi_llm_pipeline.generate_structured_json", side_effect=mocked_gen):
                res1 = run_multi_llm_pipeline(docs, db_path=db_path)
                self.assertEqual(res1["status"], "partial")
                self.assertEqual(len(res1["failed_documents"]), 1)
                self.assertEqual(res1["failed_documents"][0]["filename"], "paper_b.pdf")
                self.assertEqual(len(res1["worker1_results"]), 1)
                self.assertIsNotNone(res1["master_report"])

                # Now paper_b recovers on rerun while paper_a Worker1/Worker2 hit cache
                fail_b = False
                res2 = run_multi_llm_pipeline(docs, db_path=db_path)
                self.assertEqual(res2["status"], "completed")
                self.assertEqual(res2["failed_documents"], [])
                self.assertEqual(len(res2["worker1_results"]), 2)
                self.assertGreaterEqual(res2["cache_hits"], 2)

    def test_gemini_fallback_and_thread_safe_exhausted_set(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.models = self

            def generate_content(self, *, model, contents, config):
                if model == "gemini-3.6-flash":
                    raise genai_errors.APIError(
                        429,
                        {"error": {"message": "Quota exceeded per day", "status": "RESOURCE_EXHAUSTED"}},
                        None,
                    )
                return SimpleNamespace(
                    parsed=None,
                    text='{"filename":"a.pdf","research_objectives":["o"],"core_thesis":"t","methodology":"m","key_findings":[],"limitations":[],"key_terms":[]}',
                    usage_metadata={"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
                )

        with patch("modules.ai_analysis.resolve_api_key", return_value="fake-key"), patch(
            "modules.ai_analysis.get_gemini_client", return_value=FakeClient()
        ):
            results = []
            errs = []

            def _worker():
                try:
                    results.append(
                        generate_structured_json(
                            prompt="test",
                            schema=Worker1DocumentAnalysis,
                            system_instruction="sys",
                            model="gemini-3.6-flash",
                            allow_fallback=True,
                            retry_backoff_base=0.0,
                        )
                    )
                except Exception as exc:
                    errs.append(exc)

            threads = [threading.Thread(target=_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errs, [])
            self.assertEqual(len(results), 4)
            self.assertTrue(all(r["_model_used"] != "gemini-3.6-flash" for r in results))
            self.assertTrue(all(r["_usage"]["total_tokens"] == 15 for r in results))
            self.assertIn("gemini-3.6-flash", _get_daily_exhausted_snapshot())


if __name__ == "__main__":
    unittest.main()
