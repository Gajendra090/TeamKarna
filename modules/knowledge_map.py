"""Knowledge graph extraction, NetworkX analytics, and Plotly visualization."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Literal

import networkx as nx
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from modules.ai_analysis import AIServiceError, DEFAULT_MODEL, generate_structured_json
from modules.pdf_reader import build_corpus_context

NODE_COLORS: dict[str, str] = {
    "Document": "#38bdf8",
    "Concept": "#818cf8",
    "Method": "#34d399",
    "Research Gap": "#f87171",
}

NODE_SYMBOLS: dict[str, str] = {
    "Document": "square",
    "Concept": "circle",
    "Method": "diamond",
    "Research Gap": "triangle-up",
}

_STOPWORDS = {
    "about", "above", "across", "after", "against", "algorithm", "almost", "alone",
    "along", "already", "also", "although", "always", "among", "analysis", "another",
    "approach", "paper", "papers", "based", "because", "before", "being", "below",
    "between", "both", "could", "data", "different", "document", "during", "each",
    "either", "enough", "especially", "even", "every", "example", "existing", "figure",
    "first", "following", "found", "from", "further", "general", "given", "great",
    "however", "important", "include", "including", "large", "later", "less", "like",
    "likely", "limited", "many", "method", "methods", "model", "models", "more",
    "most", "much", "must", "neither", "often", "only", "order", "other", "others",
    "over", "page", "part", "particular", "performance", "present", "presented",
    "problem", "proposed", "provide", "rather", "recent", "related", "research",
    "result", "results", "same", "second", "section", "several", "should", "show",
    "shown", "shows", "similar", "since", "single", "small", "some", "specific",
    "state", "still", "study", "such", "system", "systems", "table", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "though",
    "three", "through", "thus", "time", "under", "used", "using", "various", "very",
    "well", "were", "what", "when", "where", "which", "while", "with", "within",
    "without", "work", "would", "your",
}


class GraphNode(BaseModel):
    id: str = Field(description="Unique concise node label.")
    node_type: Literal["Document", "Concept", "Method", "Research Gap"]
    summary: str = Field(description="One-sentence summary of this entity in the corpus.")


class GraphEdge(BaseModel):
    source: str = Field(description="Must match an existing node id.")
    target: str = Field(description="Must match an existing node id.")
    relationship: str = Field(description="Short verb phrase, e.g. 'proposes', 'shares', 'lacks'.")


class KnowledgeGraphPayload(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def extract_ai_knowledge_graph(
    documents: list[dict[str, Any]],
    analysis_result: dict[str, Any] | None = None,
    gap_report: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Use Gemini to extract an interconnected semantic knowledge graph across PDFs."""
    if not documents:
        raise AIServiceError("Upload at least one PDF before generating a knowledge map.")

    corpus_text = build_corpus_context(documents, max_chars=95000)
    filenames = [doc["filename"] for doc in documents]

    context_hints: dict[str, Any] = {}
    if analysis_result:
        context_hints["common_concepts"] = analysis_result.get("common_concepts", [])
    if gap_report:
        context_hints["top_gaps"] = [
            {"title": g.get("title"), "affected_documents": g.get("affected_documents", [])}
            for g in gap_report.get("methodological_gaps", [])[:5]
        ]

    hint_str = f"\nPRIOR FINDINGS:\n{json.dumps(context_hints, indent=2)}\n" if context_hints else ""

    system_instruction = (
        "You build concise, high-signal knowledge graphs for research comprehension. "
        "Include every uploaded filename as a 'Document' node, connect shared concepts, "
        "key methods, and notable research gaps, and keep node IDs short (2-5 words)."
    )

    prompt = (
        f"Create a knowledge graph connecting these uploaded PDF(s): {', '.join(filenames)}.\n\n"
        "Rules:\n"
        "1. Create 1 'Document' node per uploaded PDF using its exact filename as 'id'.\n"
        "2. Add 8 to 20 high-value 'Concept', 'Method', and 'Research Gap' nodes.\n"
        "3. Create clear 'edges' where both 'source' and 'target' exactly match node 'id's.\n"
        "4. Highlight cross-document bridges so shared concepts link multiple documents.\n"
        f"{hint_str}\n"
        f"DOCUMENTS:\n{corpus_text}"
    )

    payload = generate_structured_json(
        prompt=prompt,
        schema=KnowledgeGraphPayload,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.2,
        allow_fallback=allow_fallback,
    )
    payload["source_mode"] = "Gemini Semantic Graph"
    return payload


def build_structural_graph_data(
    documents: list[dict[str, Any]],
    analysis_result: dict[str, Any] | None = None,
    gap_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic local knowledge graph from analysis output or PDF text terms."""
    if not documents:
        return {"nodes": [], "edges": [], "source_mode": "Empty"}

    nodes_by_id: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, node_type: str, summary: str) -> None:
        clean_id = node_id.strip()
        if not clean_id:
            return
        if clean_id not in nodes_by_id:
            nodes_by_id[clean_id] = {
                "id": clean_id,
                "node_type": node_type,
                "summary": summary,
            }

    def add_edge(source: str, target: str, relationship: str) -> None:
        s, t = source.strip(), target.strip()
        if not s or not t or s == t:
            return
        if s not in nodes_by_id or t not in nodes_by_id:
            return
        key = (min(s, t), max(s, t), relationship)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"source": s, "target": t, "relationship": relationship})

    for doc in documents:
        add_node(
            doc["filename"],
            "Document",
            f"{doc.get('title', doc['filename'])} ({doc.get('page_count', 0)} pages, {doc.get('word_count', 0)} words)",
        )

    if analysis_result:
        for concept in analysis_result.get("common_concepts", []):
            cname = concept.get("concept", "").strip()
            if not cname:
                continue
            add_node(cname, "Concept", concept.get("explanation", "Shared corpus concept."))
            for fname in concept.get("documents", []):
                if fname in nodes_by_id:
                    add_edge(fname, cname, "discusses")

        for doc_item in analysis_result.get("document_analyses", []):
            fname = doc_item.get("filename", "").strip()
            if fname not in nodes_by_id:
                continue
            for term in doc_item.get("key_terms", [])[:4]:
                clean_term = term.strip()
                if clean_term:
                    add_node(clean_term, "Concept", f"Key term highlighted in {fname}.")
                    add_edge(fname, clean_term, "highlights")

        if gap_report:
            for gap in gap_report.get("methodological_gaps", [])[:5]:
                gtitle = gap.get("title", "").strip()
                if not gtitle:
                    continue
                add_node(gtitle, "Research Gap", gap.get("description", ""))
                for fname in gap.get("affected_documents", []):
                    if fname in nodes_by_id:
                        add_edge(fname, gtitle, "has gap")

        return {
            "nodes": list(nodes_by_id.values()),
            "edges": edges,
            "source_mode": "Analysis-Derived Structural Graph",
        }

    # Fallback: extract top domain keywords directly from PyMuPDF text
    doc_keywords: dict[str, list[tuple[str, int]]] = {}
    doc_freq: Counter[str] = Counter()

    for doc in documents:
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", doc.get("full_text", "").lower())
            if tok not in _STOPWORDS and not tok.isdigit()
        ]
        counts = Counter(tokens)
        top_terms = counts.most_common(12)
        doc_keywords[doc["filename"]] = top_terms
        for term, _ in top_terms:
            doc_freq[term] += 1

    # Prefer shared terms across docs, plus highest-frequency local terms
    for fname, term_counts in doc_keywords.items():
        ranked = sorted(
            term_counts,
            key=lambda item: (doc_freq[item[0]] > 1, doc_freq[item[0]], item[1]),
            reverse=True,
        )[:7]
        for term, freq in ranked:
            label = term.replace("-", " ").title()
            shared_note = (
                f"Appears across {doc_freq[term]} documents."
                if doc_freq[term] > 1
                else f"Frequent term in {fname} ({freq} occurrences)."
            )
            add_node(label, "Concept", shared_note)
            add_edge(fname, label, "mentions")

    return {
        "nodes": list(nodes_by_id.values()),
        "edges": edges,
        "source_mode": "Keyword Co-occurrence Graph",
    }


def build_networkx_graph(graph_data: dict[str, Any]) -> nx.Graph:
    """Convert structured node/edge data into an annotated NetworkX graph."""
    graph = nx.Graph()

    for node in graph_data.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        node_type = node.get("node_type", "Concept")
        if node_type not in NODE_COLORS:
            node_type = "Concept"
        graph.add_node(
            node_id,
            label=node_id,
            node_type=node_type,
            summary=node.get("summary", ""),
        )

    for edge in graph_data.get("edges", []):
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        rel = str(edge.get("relationship", "relates to")).strip() or "relates to"
        if source in graph and target in graph and source != target:
            graph.add_edge(source, target, relationship=rel)

    if graph.number_of_nodes() > 0:
        centrality = nx.degree_centrality(graph)
        for node_id in graph.nodes:
            graph.nodes[node_id]["degree"] = graph.degree[node_id]
            graph.nodes[node_id]["centrality"] = round(centrality.get(node_id, 0.0), 3)

    return graph


def summarize_graph_metrics(graph: nx.Graph) -> dict[str, Any]:
    """Compute real graph statistics and bridge concepts from the NetworkX graph."""
    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "density": 0.0,
            "top_concepts": [],
        }

    concept_nodes = [
        (node_id, data.get("degree", 0), data.get("node_type", "Concept"), data.get("summary", ""))
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type") != "Document"
    ]
    concept_nodes.sort(key=lambda item: item[1], reverse=True)

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "density": round(nx.density(graph), 3),
        "top_concepts": [
            {"label": c[0], "connections": c[1], "type": c[2], "summary": c[3]}
            for c in concept_nodes[:6]
        ],
    }


def render_plotly_knowledge_map(
    graph: nx.Graph,
    visible_types: list[str] | None = None,
    layout_spacing: float = 1.4,
) -> go.Figure:
    """Create an interactive dark-theme Plotly figure from a NetworkX graph."""
    if visible_types:
        kept = [
            n for n, data in graph.nodes(data=True)
            if data.get("node_type") in visible_types
        ]
        subgraph = graph.subgraph(kept).copy()
    else:
        subgraph = graph

    fig = go.Figure()

    if subgraph.number_of_nodes() == 0:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.55)",
            height=520,
            annotations=[
                dict(
                    text="No nodes match the selected filter.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="#94a3b8", size=14),
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    n_nodes = max(1, subgraph.number_of_nodes())
    k_val = layout_spacing / math.sqrt(n_nodes)
    pos = nx.spring_layout(subgraph, k=k_val, iterations=90, seed=42)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    mid_x: list[float] = []
    mid_y: list[float] = []
    mid_hover: list[str] = []

    for u, v, data in subgraph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)
        rel = data.get("relationship", "relates to")
        mid_hover.append(f"{u} —[{rel}]→ {v}")

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1.4, color="#334155"),
                hoverinfo="none",
                mode="lines",
                name="Connections",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker=dict(size=6, color="rgba(148, 163, 184, 0.25)"),
                text=mid_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    for node_type, color in NODE_COLORS.items():
        typed_nodes = [
            (n, data) for n, data in subgraph.nodes(data=True)
            if data.get("node_type") == node_type
        ]
        if not typed_nodes:
            continue

        xs, ys, labels, hovers, sizes = [], [], [], [], []
        for node_id, data in typed_nodes:
            x, y = pos[node_id]
            xs.append(x)
            ys.append(y)
            labels.append(node_id)
            deg = subgraph.degree[node_id]
            base_size = 24 if node_type == "Document" else 16
            sizes.append(min(38, base_size + deg * 2.8))
            neighbors = list(subgraph.neighbors(node_id))
            neighbor_preview = ", ".join(neighbors[:6]) + ("..." if len(neighbors) > 6 else "")
            summary = data.get("summary", "")
            hovers.append(
                f"<b>{node_id}</b><br>"
                f"Type: {node_type}<br>"
                f"Connections: {deg}<br>"
                f"{summary}<br>"
                f"<span style='color:#94a3b8'>Linked: {neighbor_preview or 'None'}</span>"
            )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=node_type,
                text=labels,
                textposition="top center",
                textfont=dict(size=11, color="#e2e8f0"),
                hovertext=hovers,
                hoverinfo="text",
                marker=dict(
                    symbol=NODE_SYMBOLS.get(node_type, "circle"),
                    size=sizes,
                    color=color,
                    line=dict(width=1.5, color="#0b1120"),
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        hovermode="closest",
        height=600,
        margin=dict(l=12, r=12, t=42, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#e2e8f0", size=12),
            bgcolor="rgba(17, 24, 39, 0.65)",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
