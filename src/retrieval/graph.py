from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion
from src.retrieval.legal_signals import is_relation_query
from src.retrieval.reranker import rerank


def build_graph(relationships: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", "")).strip()
        dst = str(row.get("other_doc_id", "")).strip()
        rel_type = str(row.get("relationship", "")).strip()
        if src and dst:
            graph.add_edge(src, dst, rel_type=rel_type)
    return graph


def _expand_with_paths(graph: nx.DiGraph, seed_ids: list[str], max_hops: int, limit: int) -> dict[str, dict[str, Any]]:
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


def _safe_int(value: Any, default: int = 10**9) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _clone_doc(doc: Document, extra_meta: dict | None = None) -> Document:
    meta = dict(doc.metadata or {})
    if extra_meta:
        meta.update(extra_meta)
    return Document(page_content=doc.page_content, metadata=meta)


def _dedupe_docs(docs: list[Document]) -> list[Document]:
    """Deduplicate by doc_id + article + chunk_index while preserving order."""
    out: list[Document] = []
    seen = set()
    for doc in docs:
        meta = doc.metadata or {}
        key = (
            str(meta.get("doc_id", "")).strip(),
            str(meta.get("article", "")).strip(),
            str(meta.get("chunk_index", "")).strip(),
            (doc.page_content or "")[:80],
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _diversify_docs(docs: list[Document], k: int, max_per_doc: int = 2) -> list[Document]:
    """Avoid one document occupying all Top-K slots in graph relation queries."""
    docs = _dedupe_docs(docs)
    out: list[Document] = []
    per_doc_count: dict[str, int] = {}

    for doc in docs:
        did = str((doc.metadata or {}).get("doc_id", "")).strip()
        if not did:
            continue
        if per_doc_count.get(did, 0) >= max_per_doc:
            continue
        out.append(doc)
        per_doc_count[did] = per_doc_count.get(did, 0) + 1
        if len(out) >= k:
            return out

    # If strict diversification returns too few docs, fill from the remaining candidates.
    seen_ids = {id(doc) for doc in out}
    for doc in docs:
        if id(doc) in seen_ids:
            continue
        out.append(doc)
        if len(out) >= k:
            break
    return out[:k]


def _fallback_docs_from_bm25(bm25, doc_ids: list[str], k: int) -> list[Document]:
    """
    Retrieve representative chunks directly from the BM25 document list by doc_id.

    This is more stable than querying the vector store with an empty query. For graph
    relations, once the graph has identified a related doc_id, we should preserve at
    least one representative chunk of that document, even if its text is not very
    similar to the current query.
    """
    all_docs = list(getattr(bm25, "documents", []) or [])
    if not all_docs:
        return []

    wanted = {str(x).strip() for x in doc_ids if str(x).strip()}
    by_doc: dict[str, list[Document]] = {doc_id: [] for doc_id in wanted}

    for doc in all_docs:
        did = str((doc.metadata or {}).get("doc_id", "")).strip()
        if did in by_doc:
            by_doc[did].append(doc)

    out: list[Document] = []
    per_doc_k = max(1, min(2, k // max(len(wanted), 1) + 1))

    for doc_id in doc_ids:
        did = str(doc_id).strip()
        candidates = by_doc.get(did, [])
        if not candidates:
            continue

        def sort_key(doc: Document):
            meta = doc.metadata or {}
            article = str(meta.get("article", "")).strip()
            # Prefer header/early chunks, then Article 1, then other low-index chunks.
            article_rank = 0 if article == "" else (1 if article == "1" else 2)
            chunk_index = _safe_int(meta.get("chunk_index", meta.get("chunk_id", 10**9)))
            content_len = len(doc.page_content or "")
            return (article_rank, chunk_index, -content_len)

        for doc in sorted(candidates, key=sort_key)[:per_doc_k]:
            out.append(_clone_doc(doc))
            if len(out) >= k:
                return _dedupe_docs(out)[:k]

    return _dedupe_docs(out)[:k]


def _retrieve_by_doc_ids(store, bm25, query: str, doc_ids: list[str], k: int) -> list[Document]:
    if not doc_ids:
        return []

    docs: list[Document] = []

    # First try vector retrieval with a hard doc_id filter.
    try:
        docs = dense_search(store, query, k=k, metadata_filter={"doc_id": {"$in": doc_ids}})
    except Exception:
        docs = []

    # If vector search misses some graph-related doc_ids, add representative chunks
    # directly from the indexed BM25 documents.
    got_doc_ids = {str((doc.metadata or {}).get("doc_id", "")).strip() for doc in docs}
    missing_ids = [doc_id for doc_id in doc_ids if str(doc_id).strip() and str(doc_id).strip() not in got_doc_ids]

    if missing_ids:
        docs.extend(_fallback_docs_from_bm25(bm25, missing_ids, k=max(k, len(missing_ids))))

    # If vector search returned nothing, try per-doc filtered dense search as a last resort.
    if not docs:
        per_doc_k = max(1, min(3, k // max(len(doc_ids), 1) + 1))
        for doc_id in doc_ids[: min(len(doc_ids), int(getattr(config.retrieval, "graph_docid_filter_limit", 40)))]:
            try:
                docs.extend(dense_search(store, query, k=per_doc_k, metadata_filter={"doc_id": doc_id}))
            except Exception:
                continue

    return _dedupe_docs(docs)[:k]


def _unique_seed_ids(seed_docs: list[Document], limit: int) -> list[str]:
    ids: list[str] = []
    seen = set()
    for doc in seed_docs:
        did = str((doc.metadata or {}).get("doc_id", "")).strip()
        if did and did not in seen:
            ids.append(did)
            seen.add(did)
        if len(ids) >= limit:
            break
    return ids


def graph_search(store, bm25_or_graph, graph_or_query=None, query: str | None = None, k: int = 5, max_hops: int | None = None) -> list[Document]:
    """Graph-guided retrieval.

    Supports both signatures:
        graph_search(store, bm25, graph, query, k=5)
        graph_search(store, graph, query, k=5)
    """
    if query is None:
        graph = bm25_or_graph
        query = graph_or_query
        from src.indexing.bm25_index import load_bm25_index
        bm25 = load_bm25_index()
    else:
        bm25 = bm25_or_graph
        graph = graph_or_query

    max_hops = int(max_hops or getattr(config.retrieval, "graph_max_hops", 1))
    expansion_limit = int(getattr(config.retrieval, "graph_expansion_limit", 40))
    seed_doc_limit = int(getattr(config.retrieval, "graph_seed_doc_limit", 8))

    pool_k = max(
        k * int(getattr(config.retrieval, "pool_multiplier", 6)),
        int(getattr(config.retrieval, "min_pool_size", 30)),
    )
    seed_docs = hybrid_search(store, bm25, query, k=pool_k)
    seed_ids = _unique_seed_ids(seed_docs, limit=seed_doc_limit)

    relation_query = is_relation_query(query)

    if not seed_ids or graph is None:
        if relation_query:
            return _diversify_docs(seed_docs, k=k, max_per_doc=2)
        return rerank(query, seed_docs, k=k, force=True)

    expanded_info = _expand_with_paths(graph, seed_ids, max_hops=max_hops, limit=expansion_limit)
    expanded_ids = list(expanded_info.keys())
    extra_docs = _retrieve_by_doc_ids(store, bm25, query, expanded_ids, k=pool_k)

    tagged_extra_docs: list[Document] = []
    for doc in extra_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", "")).strip()
        info = expanded_info.get(doc_id)
        if info:
            tagged_extra_docs.append(
                _clone_doc(
                    doc,
                    {
                        "graph_distance": info["distance"],
                        "graph_path": info["path"],
                        "graph_source": "expanded",
                    },
                )
            )
        else:
            tagged_extra_docs.append(_clone_doc(doc, {"graph_source": "expanded"}))
    extra_docs = tagged_extra_docs

    if relation_query:
        fused = reciprocal_rank_fusion(
            [extra_docs, seed_docs],
            rrf_k=int(getattr(config.retrieval, "rrf_k", 50)),
            weights=[float(getattr(config.retrieval, "graph_weight", 1.5)), 1.0],
            query=query,
        )
        # Do not force rerank relation-query results: rerank can incorrectly drop
        # historical/replaced/amending documents whose wording differs from the query.
        return _diversify_docs(fused, k=k, max_per_doc=2)

    fused = reciprocal_rank_fusion(
        [seed_docs, extra_docs],
        rrf_k=int(getattr(config.retrieval, "rrf_k", 50)),
        weights=[1.0, 0.55],
        query=query,
    )
    return rerank(query, fused, k=k, force=True)
