"""Deterministic 3-PDF test corpus and 100-question dataset for AI Ambiguity Detector evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf

from modules.pdf_reader import extract_pdf_bytes

DATASET_PATH = Path(__file__).resolve().parent / "ambiguity_dataset.json"

CORPUS_SPECS: list[dict[str, Any]] = [
    {
        "filename": "paper1_search_algorithms.pdf",
        "title": "Informed and Uninformed Graph Search Strategies",
        "pages": [
            (
                "1. Uninformed Search Algorithms\n"
                "Breadth-First Search (BFS) explores nodes level by level using a FIFO queue. "
                "For branching factor b and shallowest goal depth d, BFS has time complexity O(b^d) "
                "and space complexity O(b^d), and is optimal only when all step costs are equal.\n"
                "Depth-First Search (DFS) uses a LIFO stack with space complexity O(b*m) where m is "
                "maximum path length, time complexity O(b^m), and is neither complete in infinite "
                "state spaces nor optimal.\n"
                "Iterative Deepening Depth-First Search (IDDFS) combines BFS completeness with O(b*d) "
                "linear space complexity by running depth-limited DFS with increasing depth limit."
            ),
            (
                "2. Informed Heuristic Search and A*\n"
                "Greedy Best-First Search expands the node with lowest heuristic estimate h(n). "
                "It runs fast in practice with worst-case complexity O(b^m) but is neither complete "
                "nor optimal.\n"
                "A* Search evaluates f(n) = g(n) + h(n), where g(n) is the exact path cost from the "
                "start node to n, and h(n) is the estimated cost from n to the goal.\n"
                "Admissibility: A heuristic h(n) is admissible if it never overestimates the true "
                "minimal cost to reach the goal (0 <= h(n) <= h*(n)). Tree-search A* is optimal when "
                "h(n) is admissible. Graph-search A* requires consistency (monotonicity): "
                "h(n) <= c(n, a, n') + h(n').\n"
                "Example benchmark grid: Start node S has g(S)=0, h(S)=7. Neighbor A has step cost "
                "c(S,A)=2 and h(A)=4, giving f(A)=6. Neighbor B has step cost c(S,B)=3 and h(B)=5, "
                "giving f(B)=8. Therefore A* expands node A first."
            ),
            (
                "3. Memory-Bounded Search and Limitations\n"
                "Standard A* stores all generated nodes in the open and closed sets, causing O(b^d) "
                "exponential memory growth on large graphs. Iterative Deepening A* (IDA*) reduces "
                "memory usage to O(b*d) by pruning branches whose f(n) exceeds an f-cost threshold.\n"
                "Simplified Memory-Bounded A* (SMA*) operates within a fixed memory budget M by "
                "dropping the worst leaf node (highest f-value) when memory is full.\n"
                "Limitation: Neither A* nor IDA* handles dynamic edge weights without replanning "
                "algorithms such as D* Lite."
            ),
        ],
    },
    {
        "filename": "paper2_db_normalization.pdf",
        "title": "Relational Database Normalization and Indexing",
        "pages": [
            (
                "1. Functional Dependencies and Normal Forms\n"
                "First Normal Form (1NF) requires every attribute domain to contain only atomic, "
                "indivisible values with no repeating groups.\n"
                "Second Normal Form (2NF) requires a relation to be in 1NF and have no partial "
                "dependencies: every non-prime attribute must be fully functionally dependent on the "
                "entire candidate key.\n"
                "Third Normal Form (3NF) removes transitive dependencies: for every nontrivial "
                "functional dependency X -> A, either X is a superkey or A is a prime attribute.\n"
                "Boyce-Codd Normal Form (BCNF) is stricter than 3NF: for every nontrivial functional "
                "dependency X -> A, X must be a superkey. BCNF guarantees lossless-join decomposition "
                "but does not always preserve all functional dependencies."
            ),
            (
                "2. Indexing Structures and Query Latency\n"
                "B+ Tree indexes keep all record pointers in sorted leaf nodes linked sequentially, "
                "providing O(log_B N) lookup time and fast range queries.\n"
                "Hash indexes offer O(1) average equality lookup cost but cannot accelerate range "
                "predicates (such as BETWEEN, <, >) or prefix sorting.\n"
                "Empirical OLTP benchmark on 10 million rows: B+ Tree index achieved 1.8 ms point "
                "lookup and 14.2 ms range scan (1,000 rows). Hash index achieved 0.6 ms point lookup "
                "but fell back to full table scan (420 ms) on range scans. Each extra secondary index "
                "increased write latency by approximately 18% due to index maintenance."
            ),
            (
                "3. Transactions and Concurrency Control\n"
                "ACID properties stand for Atomicity, Consistency, Isolation, and Durability.\n"
                "Two-Phase Locking (2PL) guarantees conflict serializability via growing and shrinking "
                "lock phases, though Strict 2PL is required to prevent cascading aborts.\n"
                "Multi-Version Concurrency Control (MVCC) avoids read-write blocking by maintaining "
                "row snapshots, enabling non-blocking consistent reads at Snapshot Isolation.\n"
                "Limitation: Pure BCNF decomposition increases multi-table join overhead in analytical "
                "read-heavy workloads, so controlled denormalization is sometimes used in data warehouses."
            ),
        ],
    },
    {
        "filename": "paper3_rag_systems.pdf",
        "title": "Retrieval-Augmented Generation Evaluation and Reranking",
        "pages": [
            (
                "1. Chunking and Dense Retrieval Architecture\n"
                "Retrieval-Augmented Generation (RAG) grounds language model responses in external "
                "documents to reduce factual hallucinations.\n"
                "Fixed-size chunking splits documents into 512-token segments with a 64-token overlap "
                "to preserve boundary context. Semantic chunking groups sentences where embedding "
                "cosine similarity exceeds a 0.72 threshold.\n"
                "Dense retrieval encodes queries and chunks into 768-dimensional vectors and scores "
                "similarity using cosine distance. BM25 provides lexical sparse matching for exact "
                "acronyms, identifiers, and rare terms."
            ),
            (
                "2. Hybrid Retrieval and Cross-Encoder Reranking\n"
                "Hybrid retrieval merges BM25 sparse rankings and dense vector rankings using "
                "Reciprocal Rank Fusion (RRF): score(d) = sum(1 / (k + rank_i(d))) with smoothing "
                "constant k = 60.\n"
                "A two-stage pipeline retrieves top-50 candidates via RRF and reranks them using a "
                "Cross-Encoder to select the final top-5 context passages.\n"
                "Benchmark on 500 multi-hop QA pairs: BM25 alone reached 0.61 Recall@5; dense vector "
                "alone reached 0.74 Recall@5; Hybrid RRF reached 0.81 Recall@5; Hybrid RRF + Cross-Encoder "
                "reranking reached 0.89 Recall@5 at an additional 85 ms latency per query."
            ),
            (
                "3. Groundedness, Hallucination Rate, and Limitations\n"
                "Hallucination rate dropped from 24.5% (zero-retrieval LLM baseline) to 4.2% when "
                "enforcing strict citation prompts over reranked top-5 passages.\n"
                "Failure modes: When retrieval recall fails or top-5 chunks exceed the context budget, "
                "models may still omit edge-case caveats. Cross-Encoder reranking adds GPU compute cost "
                "and is unsuitable for sub-20 ms latency budgets without distillation."
            ),
        ],
    },
]


def build_benchmark_corpus(save_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Build the 3-PDF multi-document benchmark corpus in memory (or save_dir)."""
    docs: list[dict[str, Any]] = []
    for spec in CORPUS_SPECS:
        pdf = pymupdf.open()
        for page_text in spec["pages"]:
            page = pdf.new_page(width=595, height=842)
            tw = pymupdf.TextWriter(page.rect)
            tw.fill_textbox(pymupdf.Rect(40, 40, 555, 802), page_text, fontsize=11)
            tw.write_text(page)
        pdf.set_metadata({"title": spec["title"], "author": "KNOWLEDGE X Benchmark"})
        raw_bytes = pdf.tobytes()
        pdf.close()
        record = extract_pdf_bytes(raw_bytes, spec["filename"], save_dir=save_dir)
        docs.append(record)
    return docs


def build_dataset_records() -> list[dict[str, Any]]:
    """Construct exactly 30 Clear, 40 Ambiguous, and 30 Context-Dependent questions."""
    clear_items: list[dict[str, Any]] = [
        # --- 30 Clear Questions ---
        {"id": "CLR-01", "category": "Clear", "question": "In paper1_search_algorithms.pdf, what data structure does Breadth-First Search (BFS) use?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["fifo", "queue"]},
        {"id": "CLR-02", "category": "Clear", "question": "What are the time and space complexities of Breadth-First Search (BFS) in paper1_search_algorithms.pdf?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b^d"]},
        {"id": "CLR-03", "category": "Clear", "question": "Under what condition is Breadth-First Search optimal according to paper1_search_algorithms.pdf?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["step cost", "equal"]},
        {"id": "CLR-04", "category": "Clear", "question": "What is the space complexity of Depth-First Search (DFS) in paper1_search_algorithms.pdf?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*m", "bm"]},
        {"id": "CLR-05", "category": "Clear", "question": "How does Iterative Deepening Depth-First Search (IDDFS) reduce memory usage compared to BFS?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*d", "depth"]},
        {"id": "CLR-06", "category": "Clear", "question": "What formula does A* Search use to evaluate f(n) in paper1_search_algorithms.pdf?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["g(n)", "h(n)"]},
        {"id": "CLR-07", "category": "Clear", "question": "What defines an admissible heuristic h(n) in paper1_search_algorithms.pdf?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["overestimate"]},
        {"id": "CLR-08", "category": "Clear", "question": "What additional property beyond admissibility does graph-search A* require for optimality?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["consistency", "monotonicity"]},
        {"id": "CLR-09", "category": "Clear", "question": "In the example benchmark grid in paper1_search_algorithms.pdf, why does A* expand neighbor A before neighbor B?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["6", "8"]},
        {"id": "CLR-10", "category": "Clear", "question": "How does Simplified Memory-Bounded A* (SMA*) handle running out of memory?", "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["leaf", "highest"]},
        {"id": "CLR-11", "category": "Clear", "question": "What requirement must values satisfy for a relation to be in First Normal Form (1NF)?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["atomic"]},
        {"id": "CLR-12", "category": "Clear", "question": "What type of functional dependency is eliminated by Second Normal Form (2NF)?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["partial"]},
        {"id": "CLR-13", "category": "Clear", "question": "How does Third Normal Form (3NF) handle transitive dependencies in paper2_db_normalization.pdf?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["superkey", "prime"]},
        {"id": "CLR-14", "category": "Clear", "question": "What is the key difference between 3NF and Boyce-Codd Normal Form (BCNF) in paper2_db_normalization.pdf?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["superkey"]},
        {"id": "CLR-15", "category": "Clear", "question": "In paper2_db_normalization.pdf, why can Hash indexes not accelerate SQL BETWEEN range predicates?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["range"]},
        {"id": "CLR-16", "category": "Clear", "question": "What point lookup latency did the B+ Tree index achieve on the 10-million-row OLTP benchmark?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["1.8"]},
        {"id": "CLR-17", "category": "Clear", "question": "By what percentage did each extra secondary index increase write latency in the OLTP benchmark?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["18"]},
        {"id": "CLR-18", "category": "Clear", "question": "What do the four letters in ACID stand for in paper2_db_normalization.pdf?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["atomicity", "consistency", "isolation", "durability"]},
        {"id": "CLR-19", "category": "Clear", "question": "How does Multi-Version Concurrency Control (MVCC) avoid read-write blocking?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["snapshot"]},
        {"id": "CLR-20", "category": "Clear", "question": "Why is controlled denormalization sometimes used in analytical data warehouses according to paper2_db_normalization.pdf?", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["join"]},
        {"id": "CLR-21", "category": "Clear", "question": "What chunk size and overlap are used for fixed-size chunking in paper3_rag_systems.pdf?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["512", "64"]},
        {"id": "CLR-22", "category": "Clear", "question": "What cosine similarity threshold is used for semantic chunking in paper3_rag_systems.pdf?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.72"]},
        {"id": "CLR-23", "category": "Clear", "question": "What smoothing constant k is used in the Reciprocal Rank Fusion (RRF) formula in paper3_rag_systems.pdf?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["60"]},
        {"id": "CLR-24", "category": "Clear", "question": "What Recall@5 score did Hybrid RRF with Cross-Encoder reranking achieve on the 500 multi-hop QA benchmark?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.89"]},
        {"id": "CLR-25", "category": "Clear", "question": "How much additional latency per query does Cross-Encoder reranking add in paper3_rag_systems.pdf?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["85"]},
        {"id": "CLR-26", "category": "Clear", "question": "By how much did strict citation prompts over reranked top-5 passages reduce the hallucination rate compared to the zero-retrieval baseline?", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["24.5", "4.2"]},
        {"id": "CLR-27", "category": "Clear", "question": "Compare the 10-million-row OLTP point lookup latency of B+ Tree versus Hash indexes in paper2_db_normalization.pdf.", "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["1.8", "0.6"]},
        {"id": "CLR-28", "category": "Clear", "question": "Compare the Recall@5 scores of BM25 alone versus Hybrid RRF alone in paper3_rag_systems.pdf.", "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.61", "0.81"]},
        # 2 clear conversational follow-up questions with explicit prior turn
        {"id": "CLR-29", "category": "Clear", "question": "What is its space complexity?", "chat_history": [{"role": "user", "content": "Explain Iterative Deepening A* (IDA*) from paper1_search_algorithms.pdf."}], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*d"]},
        {"id": "CLR-30", "category": "Clear", "question": "How many dimensions do those vectors have?", "chat_history": [{"role": "user", "content": "How does dense retrieval encode queries and chunks in paper3_rag_systems.pdf?"}], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["768"]},
    ]

    ambiguous_items: list[dict[str, Any]] = [
        {"id": "AMB-01", "category": "Ambiguous", "question": "Which algorithm is better?", "expected_unclear_hint": "better", "simulated_clarification": "Compare A* Search vs Greedy Best-First Search for optimality in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["a*", "greedy", "optimal"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["optimal"]},
        {"id": "AMB-02", "category": "Ambiguous", "question": "Which index is faster?", "expected_unclear_hint": "faster", "simulated_clarification": "Compare B+ Tree vs Hash index for 1,000-row range scans in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["b+ tree", "hash", "range"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["14.2", "420"]},
        {"id": "AMB-03", "category": "Ambiguous", "question": "Explain the threshold.", "expected_unclear_hint": "threshold", "simulated_clarification": "I mean the cosine similarity threshold used for semantic chunking in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["semantic chunking", "threshold"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.72"]},
        {"id": "AMB-04", "category": "Ambiguous", "question": "Compare the two approaches.", "expected_unclear_hint": "two approaches", "simulated_clarification": "Compare Strict 2PL vs MVCC concurrency control in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["2pl", "mvcc"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["snapshot", "lock"]},
        {"id": "AMB-05", "category": "Ambiguous", "question": "How does it handle memory limits?", "expected_unclear_hint": "it", "simulated_clarification": "I am asking about Simplified Memory-Bounded A* (SMA*) in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["sma*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["leaf", "highest"]},
        {"id": "AMB-06", "category": "Ambiguous", "question": "What is the main limitation?", "expected_unclear_hint": "limitation", "simulated_clarification": "What is the limitation of BCNF decomposition in analytical workloads in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["bcnf", "limitation"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["join"]},
        {"id": "AMB-07", "category": "Ambiguous", "question": "Explain the third form.", "expected_unclear_hint": "third form", "simulated_clarification": "I mean Third Normal Form (3NF) in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["3nf", "third normal form"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["transitive"]},
        {"id": "AMB-08", "category": "Ambiguous", "question": "How much overhead does it add?", "expected_unclear_hint": "it", "simulated_clarification": "How much query latency overhead does Cross-Encoder reranking add in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["cross-encoder", "latency"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["85"]},
        {"id": "AMB-09", "category": "Ambiguous", "question": "What happens when the heuristic is wrong?", "expected_unclear_hint": "wrong", "simulated_clarification": "In paper1_search_algorithms.pdf, what happens if h(n) is not admissible (overestimates) in A*?", "expected_resolved_keywords": ["admissible", "a*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["optimal"]},
        {"id": "AMB-10", "category": "Ambiguous", "question": "Which retrieval method works best?", "expected_unclear_hint": "best", "simulated_clarification": "Which retrieval method had the highest Recall@5 on the 500 multi-hop QA benchmark in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["recall@5"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.89", "reranking"]},
        {"id": "AMB-11", "category": "Ambiguous", "question": "What is the complexity of search?", "expected_unclear_hint": "search", "simulated_clarification": "What is the space complexity of Iterative Deepening DFS (IDDFS) in paper1_search_algorithms.pdf?", "expected_resolved_keywords": ["iddfs", "space"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*d"]},
        {"id": "AMB-12", "category": "Ambiguous", "question": "Tell me about the benchmark.", "expected_unclear_hint": "benchmark", "simulated_clarification": "Summarize the 10-million-row OLTP index benchmark in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["oltp", "10 million"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["1.8", "0.6"]},
        {"id": "AMB-13", "category": "Ambiguous", "question": "Why does performance drop?", "expected_unclear_hint": "performance", "simulated_clarification": "Why does write latency increase when adding secondary indexes in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["secondary index", "write"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["18"]},
        {"id": "AMB-14", "category": "Ambiguous", "question": "Explain the fusion formula.", "expected_unclear_hint": "fusion formula", "simulated_clarification": "Explain the Reciprocal Rank Fusion (RRF) equation and constant k in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["reciprocal rank fusion", "rrf"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["60"]},
        {"id": "AMB-15", "category": "Ambiguous", "question": "How is consistency defined here?", "expected_unclear_hint": "consistency", "simulated_clarification": "I mean heuristic consistency (monotonicity) for graph-search A* in paper1_search_algorithms.pdf, not ACID consistency.", "expected_resolved_keywords": ["consistency", "a*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["monotonicity"]},
        {"id": "AMB-16", "category": "Ambiguous", "question": "What is the tradeoff?", "expected_unclear_hint": "tradeoff", "simulated_clarification": "What is the tradeoff between 3NF and BCNF in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["3nf", "bcnf"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["lossless", "dependencies"]},
        {"id": "AMB-17", "category": "Ambiguous", "question": "How does chunking work?", "expected_unclear_hint": "chunking", "simulated_clarification": "Compare fixed-size chunking vs semantic chunking in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["fixed-size", "semantic"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["512", "0.72"]},
        {"id": "AMB-18", "category": "Ambiguous", "question": "Which variant saves more memory?", "expected_unclear_hint": "variant", "simulated_clarification": "Compare standard A* vs Iterative Deepening A* (IDA*) space complexity in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["a*", "ida*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*d"]},
        {"id": "AMB-19", "category": "Ambiguous", "question": "What does it guarantee?", "expected_unclear_hint": "it", "simulated_clarification": "What does Two-Phase Locking (2PL) guarantee in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["two-phase locking", "2pl"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["serializability"]},
        {"id": "AMB-20", "category": "Ambiguous", "question": "Why is the baseline weak?", "expected_unclear_hint": "baseline", "simulated_clarification": "What was the hallucination rate of the zero-retrieval LLM baseline in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["zero-retrieval", "hallucination"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["24.5"]},
        {"id": "AMB-21", "category": "Ambiguous", "question": "Explain the hybrid strategy.", "expected_unclear_hint": "hybrid", "simulated_clarification": "Explain how hybrid retrieval combines BM25 and dense vectors in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["bm25", "dense"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["rrf", "60"]},
        {"id": "AMB-22", "category": "Ambiguous", "question": "How do they prevent cascading issues?", "expected_unclear_hint": "they", "simulated_clarification": "Which locking protocol prevents cascading aborts in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["cascading abort"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["strict 2pl"]},
        {"id": "AMB-23", "category": "Ambiguous", "question": "What fails with dynamic changes?", "expected_unclear_hint": "dynamic changes", "simulated_clarification": "In paper1_search_algorithms.pdf, how do A* and IDA* behave when edge weights change dynamically?", "expected_resolved_keywords": ["dynamic", "a*", "ida*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["d* lite", "replanning"]},
        {"id": "AMB-24", "category": "Ambiguous", "question": "What are the numbers for the two stages?", "expected_unclear_hint": "two stages", "simulated_clarification": "How many candidates are retrieved in stage 1 and kept after Cross-Encoder reranking in stage 2 in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["rerank", "stage"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["50", "5"]},
        {"id": "AMB-25", "category": "Ambiguous", "question": "Which normal form is better?", "expected_unclear_hint": "better", "simulated_clarification": "Compare 3NF vs BCNF regarding dependency preservation vs redundancy in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["3nf", "bcnf"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["superkey"]},
        {"id": "AMB-26", "category": "Ambiguous", "question": "What is the cost for B?", "expected_unclear_hint": "B", "simulated_clarification": "In paper1_search_algorithms.pdf, what are g, h, and f(B) for neighbor B in the benchmark grid?", "expected_resolved_keywords": ["neighbor b", "f(b)"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["3", "5", "8"]},
        {"id": "AMB-27", "category": "Ambiguous", "question": "How does this approach reduce errors?", "expected_unclear_hint": "this approach", "simulated_clarification": "How does RAG with strict citation prompts over reranked top-5 passages reduce hallucination rate in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["rag", "hallucination"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["4.2"]},
        {"id": "AMB-28", "category": "Ambiguous", "question": "What is the key difference between the two search methods?", "expected_unclear_hint": "two search methods", "simulated_clarification": "Compare Breadth-First Search (BFS) vs Depth-First Search (DFS) space complexity and completeness in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["bfs", "dfs"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b^d", "b*m"]},
        {"id": "AMB-29", "category": "Ambiguous", "question": "Why does blocking occur?", "expected_unclear_hint": "blocking", "simulated_clarification": "Explain how MVCC avoids read-write blocking compared to locking in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["mvcc", "blocking"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["snapshot"]},
        {"id": "AMB-30", "category": "Ambiguous", "question": "Which method should I compare against dense retrieval?", "expected_unclear_hint": "method", "simulated_clarification": "Compare BM25 sparse retrieval vs dense retrieval in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["bm25", "dense"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.61", "0.74"]},
        {"id": "AMB-31", "category": "Ambiguous", "question": "Explain the overlap.", "expected_unclear_hint": "overlap", "simulated_clarification": "What overlap size is used in fixed-size chunking in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["chunking", "overlap"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["64"]},
        {"id": "AMB-32", "category": "Ambiguous", "question": "Which paper covers scalability limits?", "expected_unclear_hint": "scalability limits", "simulated_clarification": "Summarize the memory scalability limit of standard A* in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["a*", "memory"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b^d"]},
        {"id": "AMB-33", "category": "Ambiguous", "question": "What does the first one require?", "expected_unclear_hint": "first one", "simulated_clarification": "What does First Normal Form (1NF) require in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["1nf", "first normal form"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["atomic"]},
        {"id": "AMB-34", "category": "Ambiguous", "question": "Why is it not optimal?", "expected_unclear_hint": "it", "simulated_clarification": "Why is Greedy Best-First Search not optimal in paper1_search_algorithms.pdf?", "expected_resolved_keywords": ["greedy best-first"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["h(n)"]},
        {"id": "AMB-35", "category": "Ambiguous", "question": "How is distance measured?", "expected_unclear_hint": "distance", "simulated_clarification": "What similarity metric does dense retrieval use in paper3_rag_systems.pdf?", "expected_resolved_keywords": ["dense retrieval"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["cosine"]},
        {"id": "AMB-36", "category": "Ambiguous", "question": "What happens to writes?", "expected_unclear_hint": "writes", "simulated_clarification": "What happens to write latency when secondary indexes are added in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["write latency", "secondary"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["18"]},
        {"id": "AMB-37", "category": "Ambiguous", "question": "How does pruning work there?", "expected_unclear_hint": "pruning", "simulated_clarification": "How does IDA* prune branches in paper1_search_algorithms.pdf?", "expected_resolved_keywords": ["ida*", "prune"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["f-cost", "threshold"]},
        {"id": "AMB-38", "category": "Ambiguous", "question": "What are the failure modes?", "expected_unclear_hint": "failure modes", "simulated_clarification": "What RAG failure modes are described in Section 3 of paper3_rag_systems.pdf?", "expected_resolved_keywords": ["rag", "failure"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["recall", "context"]},
        {"id": "AMB-39", "category": "Ambiguous", "question": "Explain the isolation level.", "expected_unclear_hint": "isolation level", "simulated_clarification": "Which isolation level does MVCC enable for non-blocking reads in paper2_db_normalization.pdf?", "expected_resolved_keywords": ["mvcc", "isolation"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["snapshot isolation"]},
        {"id": "AMB-40", "category": "Ambiguous", "question": "Compare the memory bounds.", "expected_unclear_hint": "memory bounds", "simulated_clarification": "Compare how IDA* and SMA* manage memory in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["ida*", "sma*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["b*d", "leaf"]},
    ]

    context_items: list[dict[str, Any]] = [
        {"id": "CTX-01", "category": "Insufficient Context", "question": "Which index should I choose for my database table?", "simulated_clarification": "My queries run frequent SQL BETWEEN range scans on 10 million rows in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["range", "b+ tree", "hash"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["b+ tree", "14.2"]},
        {"id": "CTX-02", "category": "Insufficient Context", "question": "Can we deploy Cross-Encoder reranking under our latency budget?", "simulated_clarification": "Our latency budget is strictly sub-20 ms per query without distillation, referring to paper3_rag_systems.pdf.", "expected_resolved_keywords": ["20 ms", "cross-encoder"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["85", "unsuitable"]},
        {"id": "CTX-03", "category": "Insufficient Context", "question": "Should I normalize my schema to BCNF or keep some denormalization?", "simulated_clarification": "Our workload is read-heavy analytical data warehousing where multi-table join overhead hurts query speed, per paper2_db_normalization.pdf.", "expected_resolved_keywords": ["analytical", "denormalization", "bcnf"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["join", "denormalization"]},
        {"id": "CTX-04", "category": "Insufficient Context", "question": "Compute the f(n) value for the node.", "simulated_clarification": "Compute f(n) for Neighbor A in the example benchmark grid of paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["neighbor a", "f("], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["2", "4", "6"]},
        {"id": "CTX-05", "category": "Insufficient Context", "question": "Which search algorithm fits my memory constraint?", "simulated_clarification": "I have a strict fixed memory budget M and want an A*-style search from paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["fixed memory", "m"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["sma*"]},
        {"id": "CTX-06", "category": "Insufficient Context", "question": "Is BFS guaranteed to return the cheapest path for my graph?", "simulated_clarification": "My graph has varying positive edge weights rather than equal step costs, based on paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["bfs", "varying"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["equal"]},
        {"id": "CTX-07", "category": "Insufficient Context", "question": "Will standard A* work without replanning for my robot?", "simulated_clarification": "The environment has dynamic edge weights that change during execution, per paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["dynamic", "a*"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["d* lite"]},
        {"id": "CTX-08", "category": "Insufficient Context", "question": "Does this satisfy Third Normal Form?", "simulated_clarification": "Assume a relation in 2NF has a nontrivial functional dependency X -> A where X is not a superkey and A is a non-prime attribute, per paper2_db_normalization.pdf.", "expected_resolved_keywords": ["3nf", "superkey", "prime"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["superkey", "prime"]},
        {"id": "CTX-09", "category": "Insufficient Context", "question": "How much will my write latency change if I add indexes?", "simulated_clarification": "I plan to add one extra secondary index on an OLTP table like the benchmark in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["secondary index", "write latency"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["18"]},
        {"id": "CTX-10", "category": "Insufficient Context", "question": "Which retrieval pipeline should I pick for our QA system?", "simulated_clarification": "We need the highest Recall@5 and can afford 85 ms extra latency per query, based on paper3_rag_systems.pdf.", "expected_resolved_keywords": ["recall@5", "85"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["hybrid rrf", "cross-encoder", "0.89"]},
        {"id": "CTX-11", "category": "Insufficient Context", "question": "Is Depth-First Search safe to run on my state space?", "simulated_clarification": "My state space can contain infinite paths, evaluated using paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["dfs", "infinite"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["complete"]},
        {"id": "CTX-12", "category": "Insufficient Context", "question": "Should we use BM25 or dense retrieval for our corpus?", "simulated_clarification": "Our queries frequently search for exact acronyms and rare identifiers, per paper3_rag_systems.pdf.", "expected_resolved_keywords": ["bm25", "acronym"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["bm25"]},
        {"id": "CTX-13", "category": "Insufficient Context", "question": "Will 2PL prevent cascading aborts in our scheduler?", "simulated_clarification": "We are using basic Two-Phase Locking rather than Strict 2PL in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["2pl", "cascading"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["strict 2pl"]},
        {"id": "CTX-14", "category": "Insufficient Context", "question": "Is tree-search A* optimal for our heuristic?", "simulated_clarification": "Our heuristic h(n) can sometimes overestimate the true cost to the goal, based on paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["h(n)", "overestimate"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["admissible"]},
        {"id": "CTX-15", "category": "Insufficient Context", "question": "What happens if my relation still has partial dependencies?", "simulated_clarification": "The relation is currently in 1NF and has non-prime attributes partially dependent on a composite candidate key, per paper2_db_normalization.pdf.", "expected_resolved_keywords": ["2nf", "partial"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["2nf"]},
        {"id": "CTX-16", "category": "Insufficient Context", "question": "Does graph-search A* guarantee optimality with this heuristic?", "simulated_clarification": "The heuristic is admissible but does not satisfy consistency (monotonicity) in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["graph-search", "consistency"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["consistency"]},
        {"id": "CTX-17", "category": "Insufficient Context", "question": "Should I use Hash indexing for my workload?", "simulated_clarification": "My workload mostly does exact equality lookups rather than range queries on 10 million rows in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["hash", "equality"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["0.6", "o(1)"]},
        {"id": "CTX-18", "category": "Insufficient Context", "question": "How many chunks should we pass after retrieval?", "simulated_clarification": "We are following the two-stage RRF + Cross-Encoder setup in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["two-stage", "rerank"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["50", "5"]},
        {"id": "CTX-19", "category": "Insufficient Context", "question": "Will BCNF preserve all of our functional dependencies?", "simulated_clarification": "Check the formal guarantee of BCNF decomposition in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["bcnf", "dependencies"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["lossless"]},
        {"id": "CTX-20", "category": "Insufficient Context", "question": "What is the expected hallucination rate in our setup?", "simulated_clarification": "We use no retrieval at all (zero-retrieval LLM baseline) from paper3_rag_systems.pdf.", "expected_resolved_keywords": ["zero-retrieval", "hallucination"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["24.5"]},
        {"id": "CTX-21", "category": "Insufficient Context", "question": "Can readers block writers in our concurrency setup?", "simulated_clarification": "We use MVCC with Snapshot Isolation as described in paper2_db_normalization.pdf.", "expected_resolved_keywords": ["mvcc", "blocking"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["non-blocking", "snapshot"]},
        {"id": "CTX-22", "category": "Insufficient Context", "question": "Which algorithm gives BFS completeness without exponential space?", "simulated_clarification": "Looking for an uninformed search algorithm in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["uninformed", "completeness", "space"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["iddfs"]},
        {"id": "CTX-23", "category": "Insufficient Context", "question": "Should we rely on fixed chunking or semantic chunking?", "simulated_clarification": "We want sentence grouping based on embedding cosine similarity >= 0.72 as in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["semantic chunking", "0.72"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["semantic"]},
        {"id": "CTX-24", "category": "Insufficient Context", "question": "How long will a range scan take under my index choice?", "simulated_clarification": "Suppose we chose a Hash index on a 10-million-row table and run a 1,000-row range scan per paper2_db_normalization.pdf.", "expected_resolved_keywords": ["hash", "range scan"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["420"]},
        {"id": "CTX-25", "category": "Insufficient Context", "question": "Is Greedy Best-First Search suitable if I need optimal paths?", "simulated_clarification": "Using the properties of Greedy Best-First Search in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["greedy best-first", "optimal"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["neither complete", "not"]},
        {"id": "CTX-26", "category": "Insufficient Context", "question": "What Recall@5 should I expect from my retriever?", "simulated_clarification": "We only use dense vector retrieval without BM25 or reranking, per paper3_rag_systems.pdf.", "expected_resolved_keywords": ["dense", "recall@5"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["0.74"]},
        {"id": "CTX-27", "category": "Insufficient Context", "question": "Is this table already in 1NF?", "simulated_clarification": "One column in our table stores repeating groups / comma-separated lists instead of atomic values, per paper2_db_normalization.pdf.", "expected_resolved_keywords": ["1nf", "atomic"], "expected_source": "paper2_db_normalization.pdf", "expected_answer_keywords": ["atomic"]},
        {"id": "CTX-28", "category": "Insufficient Context", "question": "Will its accuracy drop if top-5 chunks exceed my context window?", "simulated_clarification": "Asking about the RAG failure modes described in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["rag", "context"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["caveats", "omit"]},
        {"id": "CTX-29", "category": "Insufficient Context", "question": "What is the f-value at the start node?", "simulated_clarification": "Look up the start node S in the example benchmark grid in paper1_search_algorithms.pdf.", "expected_resolved_keywords": ["start node s"], "expected_source": "paper1_search_algorithms.pdf", "expected_answer_keywords": ["7"]},
        {"id": "CTX-30", "category": "Insufficient Context", "question": "Can I combine sparse and dense rankings without training a neural scorer?", "simulated_clarification": "Using the rank fusion method in paper3_rag_systems.pdf.", "expected_resolved_keywords": ["rrf", "sparse", "dense"], "expected_source": "paper3_rag_systems.pdf", "expected_answer_keywords": ["reciprocal rank fusion", "60"]},
    ]

    dataset = clear_items + ambiguous_items + context_items
    assert len(clear_items) == 30
    assert len(ambiguous_items) == 40
    assert len(context_items) == 30
    assert len(dataset) == 100
    return dataset


def save_dataset(path: Path | str = DATASET_PATH) -> Path:
    records = build_dataset_records()
    target = Path(path)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    out = save_dataset()
    print(f"Saved {len(build_dataset_records())} evaluation items to {out}")
