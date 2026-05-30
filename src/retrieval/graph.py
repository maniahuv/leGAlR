from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from src.retrieval.reranker import rerank

RELATION_QUERY_KEYWORDS = [
    "sửa đổi", "bổ sung", "thay thế", "bãi bỏ", "hủy bỏ", "tham chiếu", "dẫn chiếu",
    "liên quan", "hiệu lực", "hết hiệu lực", "còn hiệu lực", "văn bản nào", "quy định chi tiết",
]


def is_relation_query(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in RELATION_QUERY_KEYWORDS)


def build_graph(relationships: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", "")).strip()
        dst = str(row.get("other_doc_id", "")).strip()
        rel_type = str(row.get("relationship", "")).strip()
        if src and dst:
            graph.add_edge(src, dst, rel_type=rel_type)
    return graph


def _expand_with_paths(graph: nx.DiGraph, seed_ids: set[str], max_hops: int, limit: int) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    q = deque()
    for sid in seed_ids:
        q.append((sid, 0, []))

    visited = set(seed_ids)
    while q and len(info) < limit:
        node, dist, path = q.popleft()
        if dist >= max_hops:
            continue

        edges = []
        if node in graph:
            edges.extend([(node, v, graph.get_edge_data(node, v, default={}).get("rel_type", ""), "out") for v in graph.successors(node)])
            edges.extend([(u, node, graph.get_edge_data(u, node, default={}).get("rel_type", ""), "in") for u in graph.predecessors(node)])

        for src, dst, rel_type, direction in edges:
            nb = str(dst if direction == "out" else src).strip()
            if not nb or nb in visited:
                continue
            visited.add(nb)
            new_path = path + [{"src": str(src), "dst": str(dst), "relationship": rel_type, "direction": direction}]
            info[nb] = {"distance": dist + 1, "path": new_path}
            q.append((nb, dist + 1, new_path))
            if len(info) >= limit:
                break
    return info


def _retrieve_by_doc_ids(store, query: str, doc_ids: list[str], k: int) -> list[Document]:
    if not doc_ids:
        return []
    docs: list[Document] = []
    # Chroma hỗ trợ $in ở đa số version mới; fallback từng doc_id nếu version cũ lỗi.
    try:
        return dense_search(store, query, k=k, metadata_filter={"doc_id": {"$in": doc_ids}})
    except Exception:
        pass
    per_doc_k = max(1, min(3, k // max(len(doc_ids), 1) + 1))
    for doc_id in doc_ids[: min(len(doc_ids), 50)]:
        try:
            docs.extend(dense_search(store, query, k=per_doc_k, metadata_filter={"doc_id": doc_id}))
        except Exception:
            continue
    return docs[:k]


def graph_search(store, bm25_or_graph, graph_or_query=None, query: str | None = None, k: int = 5, max_hops: int | None = None) -> list[Document]:
    """Graph-guided retrieval.

    Hỗ trợ cả signature mới:
        graph_search(store, bm25, graph, query, k=5)
    và signature cũ:
        graph_search(store, graph, query, k=5)
    """
    if query is None:
        # Backward compatibility: graph_search(store, graph, query, ...)
        graph = bm25_or_graph
        query = graph_or_query
        from src.indexing.bm25_index import load_bm25_index
        bm25 = load_bm25_index()
    else:
        bm25 = bm25_or_graph
        graph = graph_or_query

    max_hops = int(max_hops or getattr(config.retrieval, "graph_max_hops", 2))
    expansion_limit = int(getattr(config.retrieval, "graph_expansion_limit", 80))

    pool_k = max(k * int(getattr(config.retrieval, "pool_multiplier", 12)), int(getattr(config.retrieval, "min_pool_size", 50)))
    seed_docs = hybrid_search(store, bm25, query, k=pool_k)
    seed_ids = {str((d.metadata or {}).get("doc_id", "")).strip() for d in seed_docs if (d.metadata or {}).get("doc_id")}
    if not seed_ids or graph is None:
        return seed_docs[:k]

    expanded_info = _expand_with_paths(graph, seed_ids, max_hops=max_hops, limit=expansion_limit)
    expanded_ids = list(expanded_info.keys())
    extra_docs = _retrieve_by_doc_ids(store, query, expanded_ids, k=pool_k)

    for doc in extra_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", "")).strip()
        info = expanded_info.get(doc_id)
        if info:
            meta = dict(doc.metadata or {})
            meta["graph_distance"] = info["distance"]
            meta["graph_path"] = info["path"]
            doc.metadata = meta

    if is_relation_query(query):
        fused = reciprocal_rank_fusion(
            [extra_docs, seed_docs],
            rrf_k=int(getattr(config.retrieval, "rrf_k", 60)),
            weights=[float(getattr(config.retrieval, "graph_weight", 1.25)), 1.0],
            query=query,
        )
        return rerank(query, fused, k=k, force=True)

    fused = reciprocal_rank_fusion([seed_docs, extra_docs], rrf_k=int(getattr(config.retrieval, "rrf_k", 60)), weights=[1.0, 0.75], query=query)
    return rerank(query, fused, k=k)
