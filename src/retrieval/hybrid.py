from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.dense import dense_search


def _chunk_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return str(meta.get("chunk_uid") or f"{meta.get('doc_id', '')}_{meta.get('chunk_index', '')}" or doc.page_content[:80])


def _metadata_boost(query: str, doc: Document) -> float:
    q = (query or "").lower()
    meta = doc.metadata or {}
    boost = 0.0

    so = str(meta.get("so_ky_hieu", "")).lower()
    if so and so in q:
        boost += 0.20

    title = str(meta.get("title", "")).lower()
    if title:
        title_tokens = {t for t in re.findall(r"\w+", title, flags=re.UNICODE) if len(t) > 2}
        q_tokens = set(re.findall(r"\w+", q, flags=re.UNICODE))
        overlap = len(title_tokens & q_tokens)
        if overlap:
            boost += min(0.10, 0.015 * overlap)

    article = str(meta.get("article", ""))
    if article and re.search(rf"\bđiều\s+{re.escape(article)}\b", q, flags=re.IGNORECASE):
        boost += 0.25

    status = str(meta.get("tinh_trang_hieu_luc", "")).lower()
    if "còn hiệu lực" in status:
        boost += 0.03
    elif "hết hiệu lực" in status:
        boost -= 0.02
    return boost


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
    query: str = "",
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    docs_by_key: dict[str, Document] = {}
    weights = weights or [1.0] * len(ranked_lists)

    for list_idx, docs in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, doc in enumerate(docs, start=1):
            key = _chunk_key(doc)
            if not key:
                continue
            scores[key] += weight / (rrf_k + rank)
            docs_by_key.setdefault(key, doc)

    for key, doc in docs_by_key.items():
        scores[key] += _metadata_boost(query, doc)

    return [docs_by_key[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def bm25_search(bm25, query: str, k: int = 5) -> list[Document]:
    old_k = getattr(bm25, "k", None)
    try:
        bm25.k = k
        return bm25.invoke(query)
    finally:
        if old_k is not None:
            bm25.k = old_k


def hybrid_search(
    store,
    bm25,
    query: str,
    k: int = 5,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    pool_k: int | None = None,
) -> list[Document]:
    pool_k = pool_k or max(k * int(getattr(config.retrieval, "pool_multiplier", 12)), int(getattr(config.retrieval, "min_pool_size", 50)))
    dense_k = dense_k or pool_k
    bm25_k = bm25_k or pool_k

    dense_docs = dense_search(store, query, k=dense_k)
    bm25_docs = bm25_search(bm25, query, k=bm25_k)

    fused = reciprocal_rank_fusion(
        [dense_docs, bm25_docs],
        rrf_k=int(getattr(config.retrieval, "rrf_k", 60)),
        weights=[float(getattr(config.retrieval, "dense_weight", 1.0)), float(getattr(config.retrieval, "bm25_weight", 1.15))],
        query=query,
    )
    return fused[:k]
