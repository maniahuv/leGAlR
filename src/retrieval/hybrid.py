from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.dense import dense_search
from src.retrieval.legal_signals import metadata_signal_score, normalize_text


_METADATA_CACHE_ATTR = "_legal_metadata_cache_v3"

_LEGAL_STOPWORDS = {
    "và", "của", "các", "theo", "về", "cho", "trong", "là", "có", "được",
    "quy", "định", "này", "đó", "một", "những", "với", "tại", "từ", "đến",
    "hỏi", "cho", "biết", "nêu", "trình", "bày",
}

_METADATA_HINTS = (
    "luật", "nghị định", "thông tư", "nghị quyết", "quyết định", "pháp lệnh",
    "văn bản", "số ký hiệu", "số hiệu", "hiệu lực", "ban hành", "hướng dẫn",
    "sửa đổi", "bổ sung", "thay thế", "bãi bỏ", "điều", "khoản",
)


def _chunk_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return str(meta.get("chunk_uid") or f"{meta.get('doc_id', '')}_{meta.get('chunk_index', '')}" or doc.page_content[:80])


def _dedupe_documents(docs: list[Document]) -> list[Document]:
    """Preserve order while removing duplicated chunks."""
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = _chunk_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _simple_tokens(text: str) -> set[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[\w/\.\-]+", text, flags=re.UNICODE)
    return {t for t in tokens if len(t) >= 2 and t not in _LEGAL_STOPWORDS}


def _extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b(19\d{2}|20\d{2})\b", text or ""))


def _metadata_query_needed(query: str) -> bool:
    """Avoid scanning metadata for every semantic query.

    The v2 implementation scanned every BM25 document on every query. With many
    chunks this makes HYBRID extremely slow. This gate keeps metadata search for
    title/document-number/article/effectiveness-style queries only.
    """
    q = normalize_text(query)
    if re.search(r"\b\d+\s*/\s*\d{4}", q):
        return True
    if re.search(r"\b(điều|khoản)\s+\d+", q, flags=re.UNICODE):
        return True
    if any(hint in q for hint in _METADATA_HINTS):
        return True
    if "hôn nhân" in q and "gia đình" in q and any(y in q for y in _extract_years(q)):
        return True
    return False


def _build_metadata_cache(bm25) -> dict[str, Any]:
    cached = getattr(bm25, _METADATA_CACHE_ATTR, None)
    if cached is not None:
        return cached

    docs = getattr(bm25, "documents", []) or []
    entries: list[dict[str, Any]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    so_index: dict[str, list[int]] = defaultdict(list)
    article_index: dict[str, list[int]] = defaultdict(list)

    for idx, doc in enumerate(docs):
        meta = doc.metadata or {}
        title = normalize_text(str(meta.get("title", "") or ""))
        so = normalize_text(str(meta.get("so_ky_hieu", "") or ""))
        loai = normalize_text(str(meta.get("loai_van_ban", "") or ""))
        linh_vuc = normalize_text(str(meta.get("linh_vuc", "") or ""))
        nganh = normalize_text(str(meta.get("nganh", "") or ""))
        article = str(meta.get("article", "") or "").strip()
        clause = str(meta.get("clause", "") or "").strip()
        status = normalize_text(str(meta.get("tinh_trang_hieu_luc", "") or ""))
        date_text = " ".join([
            str(meta.get("ngay_ban_hanh", "") or ""),
            str(meta.get("ngay_co_hieu_luc", "") or ""),
            str(meta.get("ngay_het_hieu_luc", "") or ""),
        ])

        meta_text = " ".join([title, so, loai, linh_vuc, nganh, article, clause, date_text, status])
        tokens = _simple_tokens(meta_text)
        title_tokens = _simple_tokens(title)
        so_compact = _compact(so)
        entry = {
            "doc": doc,
            "title": title,
            "title_compact": _compact(title),
            "title_tokens": title_tokens,
            "so": so,
            "so_compact": so_compact,
            "loai": loai,
            "linh_vuc": linh_vuc,
            "nganh": nganh,
            "article": article,
            "clause": clause,
            "status": status,
            "years": _extract_years(" ".join([title, so, date_text])),
            "tokens": tokens,
        }
        entries.append(entry)

        for token in tokens:
            inverted[token].append(idx)
        if so_compact:
            so_index[so_compact].append(idx)
        if article:
            article_index[article.lower()].append(idx)

    cache = {
        "entries": entries,
        "inverted": dict(inverted),
        "so_index": dict(so_index),
        "article_index": dict(article_index),
    }
    setattr(bm25, _METADATA_CACHE_ATTR, cache)
    return cache


def _metadata_entry_score(query: str, entry: dict[str, Any], q_tokens: set[str], q_compact: str, years: set[str]) -> float:
    q = normalize_text(query)
    score = 0.0

    so_compact = entry["so_compact"]
    if so_compact and so_compact in q_compact:
        score += 4.0

    title_tokens = entry["title_tokens"]
    if q_tokens and title_tokens:
        overlap = len(q_tokens & title_tokens)
        coverage_q = overlap / max(len(q_tokens), 1)
        coverage_title = overlap / max(len(title_tokens), 1)
        if coverage_q >= 0.55 or coverage_title >= 0.45:
            score += 1.8 * coverage_q + 0.8 * coverage_title
        elif overlap >= 3:
            score += 0.35 * overlap

    title_compact = entry["title_compact"]
    if title_compact and len(title_compact) >= 16:
        if title_compact in q_compact or q_compact in title_compact:
            score += 2.0

    article = entry["article"]
    clause = entry["clause"]
    if article and re.search(rf"\bđiều\s+{re.escape(article)}\b", q, flags=re.UNICODE):
        score += 2.2
    if clause and re.search(rf"\bkhoản\s+{re.escape(clause)}\b", q, flags=re.UNICODE):
        score += 0.9

    if "hôn nhân" in q and "gia đình" in q:
        if "hôn nhân" in entry["title"] and "gia đình" in entry["title"]:
            score += 1.1
        if (
            "hôn nhân" in entry["linh_vuc"] or "gia đình" in entry["linh_vuc"]
            or "hôn nhân" in entry["nganh"] or "gia đình" in entry["nganh"]
        ):
            score += 0.35

    for doc_type in ("luật", "nghị định", "thông tư", "nghị quyết", "quyết định", "pháp lệnh"):
        if doc_type in q and doc_type in entry["loai"]:
            score += 0.25
            break

    if years and (years & entry["years"]):
        score += 0.25

    if score > 0:
        status = entry["status"]
        if "còn hiệu lực" in status:
            score += 0.08
        elif "hết hiệu lực" in status:
            score -= 0.12

    return score


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

    # Keep metadata boost small at fusion stage; the heavy legal-aware scoring is done in rerank().
    fusion_metadata_weight = float(getattr(config.retrieval, "fusion_metadata_weight", 0.18))
    for key, doc in docs_by_key.items():
        scores[key] += fusion_metadata_weight * metadata_signal_score(query, doc)

    return [docs_by_key[key] for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def bm25_search(bm25, query: str, k: int = 5) -> list[Document]:
    old_k = getattr(bm25, "k", None)
    try:
        bm25.k = k
        return bm25.invoke(query)
    finally:
        if old_k is not None:
            bm25.k = old_k


def metadata_search(bm25, query: str, k: int = 5) -> list[Document]:
    """Fast metadata/title fallback channel.

    v2 scanned every BM25 document and called underthesea-heavy tokenization for
    every query, which is too slow on CPU. This version builds a small inverted
    metadata cache once, then scores only candidates that share metadata tokens
    with the query.
    """
    if k <= 0 or not _metadata_query_needed(query):
        return []

    cache = _build_metadata_cache(bm25)
    entries: list[dict[str, Any]] = cache["entries"]
    inverted: dict[str, list[int]] = cache["inverted"]
    so_index: dict[str, list[int]] = cache["so_index"]
    article_index: dict[str, list[int]] = cache["article_index"]

    q = normalize_text(query)
    q_tokens = _simple_tokens(q)
    q_compact = _compact(q)
    years = _extract_years(q)

    counter: Counter[int] = Counter()
    for token in q_tokens:
        for idx in inverted.get(token, []):
            counter[idx] += 1

    # Exact document number candidates, e.g. 52/2014/QH13.
    for raw in re.findall(r"\b\d+\s*/\s*\d{4}\s*/\s*[\w\-Đđ]+\b", q, flags=re.UNICODE):
        key = _compact(raw)
        for idx in so_index.get(key, []):
            counter[idx] += 10

    # Article candidates, useful for exact-keyword-fact and ArticleHit.
    for article in re.findall(r"\bđiều\s+(\d+[a-zA-Z]?)\b", q, flags=re.UNICODE):
        for idx in article_index.get(article.lower(), []):
            counter[idx] += 4

    if not counter:
        return []

    max_candidates = int(getattr(config.retrieval, "metadata_max_candidates", 2500))
    candidate_ids = [idx for idx, _ in counter.most_common(max_candidates)]

    scored: list[tuple[float, Document]] = []
    seen_keys: set[str] = set()
    for idx in candidate_ids:
        entry = entries[idx]
        score = _metadata_entry_score(query, entry, q_tokens, q_compact, years)
        if score <= 0.15:
            continue
        doc = entry["doc"]
        key = _chunk_key(doc)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def hybrid_search(
    store,
    bm25,
    query: str,
    k: int = 5,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    pool_k: int | None = None,
) -> list[Document]:
    pool_k = pool_k or max(
        k * int(getattr(config.retrieval, "pool_multiplier", 6)),
        int(getattr(config.retrieval, "min_pool_size", 30)),
    )
    dense_k = dense_k or pool_k
    bm25_k = bm25_k or pool_k

    dense_docs = dense_search(store, query, k=dense_k)
    bm25_docs = bm25_search(bm25, query, k=bm25_k)
    meta_docs = metadata_search(
        bm25,
        query,
        k=int(getattr(config.retrieval, "metadata_k", max(bm25_k, 30))),
    )

    ranked_lists = [dense_docs, bm25_docs]
    weights = [
        float(getattr(config.retrieval, "dense_weight", 0.9)),
        float(getattr(config.retrieval, "bm25_weight", 1.35)),
    ]
    if meta_docs:
        ranked_lists.append(meta_docs)
        weights.append(float(getattr(config.retrieval, "metadata_weight", 2.0)))

    fused = reciprocal_rank_fusion(
        ranked_lists,
        rrf_k=int(getattr(config.retrieval, "rrf_k", 50)),
        weights=weights,
        query=query,
    )

    # For title/document-number/article queries, metadata results are not merely
    # another weak RRF signal. In legal retrieval, exact metadata matches should
    # be guaranteed to survive into the final candidate pool, otherwise rerank()
    # cannot recover them. This restores metadata-title-query recall while still
    # keeping dense/BM25 fallback candidates after the exact metadata candidates.
    if meta_docs and _metadata_query_needed(query):
        fused = _dedupe_documents(meta_docs + fused)

    return fused[:k]
