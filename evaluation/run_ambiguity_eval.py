"""Run quantitative evaluation on the 100-question AI Ambiguity Detector benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.build_dataset import DATASET_PATH, build_benchmark_corpus, save_dataset
from modules.ai_analysis import AIServiceError, answer_research_question, load_env_config
from modules.ambiguity_detector import (
    analyze_question_ambiguity,
    rewrite_clarified_question,
)


def _keyword_match_ratio(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords)


def _is_fatal_daily_quota(err_msg: str) -> bool:
    upper = (err_msg or "").upper()
    return "429" in upper and ("PER DAY" in upper or "LIMIT: 0" in upper)


def _eval_one_item(
    item: dict[str, Any],
    documents: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Evaluate classification, rewriting resolution, and final grounded answer for one item."""
    q = item["question"]
    chat_hist = item.get("chat_history")
    expected_label = item["category"]

    last_err = ""
    classification_res: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            classification_res = analyze_question_ambiguity(
                question=q,
                documents=documents,
                chat_history=chat_hist,
                model=model,
                allow_fallback=True,
            )
            break
        except AIServiceError as exc:
            last_err = str(exc)
            if _is_fatal_daily_quota(last_err):
                break
            time.sleep(2.5 * (attempt + 1))

    if classification_res is None:
        return {
            "id": item["id"],
            "expected": expected_label,
            "predicted": "ERROR",
            "error": last_err,
            "resolution_ok": None,
            "answer_ok": False,
        }

    predicted = classification_res["classification"]
    resolved_question = q
    resolution_ok: bool | None = None

    # End-to-end flow: clarify & rewrite whenever detector flags Ambiguous / Insufficient Context
    if predicted in ("Ambiguous", "Insufficient Context"):
        clarification = item.get(
            "simulated_clarification",
            "Answer the original question directly using the specified document.",
        )
        rewritten = None
        for attempt in range(3):
            try:
                rewritten = rewrite_clarified_question(
                    original_question=q,
                    user_clarification=clarification,
                    classification_info=classification_res,
                    documents=documents,
                    chat_history=chat_hist,
                    model=model,
                    allow_fallback=True,
                )
                break
            except AIServiceError as exc:
                last_err = str(exc)
                if _is_fatal_daily_quota(last_err):
                    break
                time.sleep(2.5 * (attempt + 1))

        if rewritten:
            resolved_question = rewritten.get("resolved_question", "").strip() or q
            if expected_label in ("Ambiguous", "Insufficient Context"):
                res_ratio = _keyword_match_ratio(
                    resolved_question, item.get("expected_resolved_keywords", [])
                )
                resolution_ok = (
                    resolved_question.lower() != q.lower()
                    and len(resolved_question.split()) >= 5
                    and res_ratio >= 0.5
                )
        elif expected_label in ("Ambiguous", "Insufficient Context"):
            resolution_ok = False
    elif expected_label in ("Ambiguous", "Insufficient Context"):
        # False negative: detector skipped clarification, so no resolution happened
        resolution_ok = False

    # Final grounded answer against uploaded PDFs using the clear or resolved question
    qa_res = None
    for attempt in range(3):
        try:
            qa_res = answer_research_question(
                question=resolved_question,
                documents=documents,
                chat_history=chat_hist,
                model=model,
                allow_fallback=True,
            )
            break
        except AIServiceError as exc:
            last_err = str(exc)
            if _is_fatal_daily_quota(last_err):
                break
            time.sleep(2.5 * (attempt + 1))

    answer_ok = False
    answer_text = ""
    cited_files: list[str] = []
    if qa_res:
        answer_text = qa_res.get("answer", "")
        cited_files = [c.get("filename", "") for c in qa_res.get("citations", [])]
        ans_ratio = _keyword_match_ratio(
            answer_text, item.get("expected_answer_keywords", [])
        )
        expected_src = item.get("expected_source")
        src_ok = (
            (not expected_src)
            or (expected_src in cited_files)
            or (expected_src.lower() in answer_text.lower())
        )
        answer_ok = ans_ratio >= 0.5 and src_ok

    return {
        "id": item["id"],
        "expected": expected_label,
        "predicted": predicted,
        "model_used": classification_res.get("_model_used", model),
        "unclear_phrases": classification_res.get("unclear_phrases", []),
        "interpretations_count": len(classification_res.get("interpretations", [])),
        "resolved_question": resolved_question,
        "resolution_ok": resolution_ok,
        "answer_ok": answer_ok,
        "cited_files": cited_files,
        "error": last_err if (qa_res is None) else "",
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision, recall, F1, false positive rate, resolution accuracy, and answer correctness."""
    labels = ["Clear", "Ambiguous", "Insufficient Context"]
    valid = [r for r in results if r.get("predicted") in labels]
    errors = [r for r in results if r.get("predicted") not in labels]

    confusion = {exp: {pred: 0 for pred in labels} for exp in labels}
    for r in valid:
        confusion[r["expected"]][r["predicted"]] += 1

    # Binary Clarification-Needed Detection: Positive = {Ambiguous, Insufficient Context}, Negative = {Clear}
    tp = sum(1 for r in valid if r["expected"] != "Clear" and r["predicted"] != "Clear")
    fp = sum(1 for r in valid if r["expected"] == "Clear" and r["predicted"] != "Clear")
    fn = sum(1 for r in valid if r["expected"] != "Clear" and r["predicted"] == "Clear")
    tn = sum(1 for r in valid if r["expected"] == "Clear" and r["predicted"] == "Clear")

    bin_prec = tp / (tp + fp) if (tp + fp) else 0.0
    bin_rec = tp / (tp + fn) if (tp + fn) else 0.0
    bin_f1 = (2 * bin_prec * bin_rec / (bin_prec + bin_rec)) if (bin_prec + bin_rec) else 0.0
    clear_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    for lbl in labels:
        c_tp = confusion[lbl][lbl]
        c_fp = sum(confusion[other][lbl] for other in labels if other != lbl)
        c_fn = sum(confusion[lbl][other] for other in labels if other != lbl)
        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support": sum(confusion[lbl].values()),
        }

    exact_acc = sum(confusion[l][l] for l in labels) / len(valid) if valid else 0.0

    res_items = [r for r in valid if r.get("resolution_ok") is not None]
    res_acc = sum(1 for r in res_items if r["resolution_ok"]) / len(res_items) if res_items else 0.0

    ans_acc = sum(1 for r in valid if r["answer_ok"]) / len(valid) if valid else 0.0
    ans_by_cat = {}
    for lbl in labels:
        cat_rows = [r for r in valid if r["expected"] == lbl]
        ans_by_cat[lbl] = (
            round(sum(1 for r in cat_rows if r["answer_ok"]) / len(cat_rows), 4)
            if cat_rows
            else 0.0
        )

    return {
        "total_items": len(results),
        "valid_evaluated": len(valid),
        "api_errors": len(errors),
        "binary_clarification_detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1_score": round(bin_f1, 4),
            "clear_false_positive_rate": round(clear_fpr, 4),
        },
        "three_class_accuracy": round(exact_acc, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "resolution_accuracy": round(res_acc, 4),
        "resolution_evaluated_count": len(res_items),
        "final_answer_correctness": round(ans_acc, 4),
        "final_answer_correctness_by_category": ans_by_cat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI Ambiguity Detector.")
    parser.add_argument("--limit", type=int, default=0, help="Optional subset size for quick pilot.")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent evaluation workers.")
    parser.add_argument("--model", type=str, default="", help="Override Gemini model ID.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing eval_results.json.")
    args = parser.parse_args()

    save_dataset(DATASET_PATH)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit > 0:
        by_cat = {
            "Clear": [x for x in dataset if x["category"] == "Clear"],
            "Ambiguous": [x for x in dataset if x["category"] == "Ambiguous"],
            "Insufficient Context": [x for x in dataset if x["category"] == "Insufficient Context"],
        }
        per = max(1, args.limit // 3)
        dataset = by_cat["Clear"][:per] + by_cat["Ambiguous"][:per] + by_cat["Insufficient Context"][:per]

    cfg = load_env_config(".env")
    if not cfg["api_key"]:
        raise SystemExit("GEMINI_API_KEY missing in .env; cannot run live evaluation.")

    model = args.model.strip() or cfg["model"]
    out_path = Path(__file__).resolve().parent / "eval_results.json"

    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for r in prev.get("items", []):
            if r.get("predicted") in ("Clear", "Ambiguous", "Insufficient Context"):
                existing_by_id[r["id"]] = r

    pending_items = [item for item in dataset if item["id"] not in existing_by_id]
    docs = build_benchmark_corpus(save_dir=None)
    t0 = time.time()
    results = list(existing_by_id.values())

    if pending_items:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(_eval_one_item, item, docs, model): item["id"]
                for item in pending_items
            }
            for idx, fut in enumerate(as_completed(futures), start=1):
                res = fut.result()
                results.append(res)
                if idx % 10 == 0 or idx == len(pending_items):
                    results.sort(key=lambda r: r["id"])
                    summary = compute_metrics(results)
                    summary["model"] = model
                    summary["elapsed_seconds"] = round(time.time() - t0, 2)
                    out_path.write_text(
                        json.dumps({"summary": summary, "items": results}, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"Evaluated {len(results)}/{len(dataset)} total items "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )

    results.sort(key=lambda r: r["id"])
    summary = compute_metrics(results)
    summary["model"] = model
    summary["elapsed_seconds"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps({"summary": summary, "items": results}, indent=2), encoding="utf-8")
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
