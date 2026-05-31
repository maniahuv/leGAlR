from __future__ import annotations

from typing import Any


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_top_k = _unique_preserve_order(retrieved_ids)[:k]
    relevant_set = set(_unique_preserve_order(relevant_ids))
    return 1.0 if any(doc_id in relevant_set for doc_id in retrieved_top_k) else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_top_k = _unique_preserve_order(retrieved_ids)[:k]
    relevant_set = set(_unique_preserve_order(relevant_ids))
    return sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant_set = set(_unique_preserve_order(relevant_ids))
    if not relevant_set:
        return 0.0
    retrieved_top_k = _unique_preserve_order(retrieved_ids)[:k]
    return sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(_unique_preserve_order(relevant_ids))
    for index, doc_id in enumerate(_unique_preserve_order(retrieved_ids), start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def article_hit_at_k(retrieved_pairs: list[tuple[str, str]], relevant_ids: list[str], relevant_articles: list[str], k: int) -> float:
    """Stricter legal metric: doc_id must match and article must match when articles are supplied."""
    articles = {str(a).strip().lower() for a in relevant_articles if str(a).strip()}
    if not articles:
        return 0.0
    relevant_set = set(_unique_preserve_order(relevant_ids))
    for doc_id, article in retrieved_pairs[:k]:
        if str(doc_id).strip() in relevant_set and str(article).strip().lower() in articles:
            return 1.0
    return 0.0


def mean_reciprocal_rank(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(item["retrieved_ids"], item["relevant_ids"]) for item in results) / len(results)


def evaluate_retrieval_case(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
    retrieved_pairs: list[tuple[str, str]] | None = None,
    relevant_articles: list[str] | None = None,
) -> dict:
    metrics = {
        "hit_at_k": hit_at_k(retrieved_ids, relevant_ids, k),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }
    if relevant_articles:
        metrics["article_hit_at_k"] = article_hit_at_k(retrieved_pairs or [], relevant_ids, relevant_articles, k)
    return metrics


def average_metrics(metrics: list[dict]) -> dict:
    if not metrics:
        return {"hit_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "reciprocal_rank": 0.0}
    keys = sorted({key for item in metrics for key in item.keys()})
    return {key: sum(float(item.get(key, 0.0)) for item in metrics) / len(metrics) for key in keys}
