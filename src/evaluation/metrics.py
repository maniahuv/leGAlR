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


def _norm_pair(pair) -> tuple[str, str]:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return (str(pair[0]).strip(), str(pair[1]).strip().lower())
    return ("", "")


def _unique_pairs(pairs: list[tuple[str, str]] | list[list[str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for raw in pairs or []:
        pair = _norm_pair(raw)
        if pair[0] and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _is_doc_level_pair(pair: tuple[str, str]) -> bool:
    return pair[1] in {"", "*", "__full_doc__"}


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


def article_hit_at_k(
    retrieved_pairs: list[tuple[str, str]],
    relevant_ids: list[str],
    relevant_articles: list[str],
    k: int,
) -> float:
    """Loose article metric: doc_id is in relevant_ids and article is in relevant_articles."""
    articles = {str(a).strip().lower() for a in relevant_articles if str(a).strip()}
    if not articles:
        return 0.0
    relevant_set = set(_unique_preserve_order(relevant_ids))
    for doc_id, article in _unique_pairs(retrieved_pairs)[:k]:
        if str(doc_id).strip() in relevant_set and str(article).strip().lower() in articles:
            return 1.0
    return 0.0


def article_recall_at_k(
    retrieved_pairs: list[tuple[str, str]],
    relevant_ids: list[str],
    relevant_articles: list[str],
    k: int,
) -> float:
    """Loose recall over expected article numbers within relevant documents.

    This is intentionally loose because many old test cases store articles
    separately from doc_ids rather than as strict (doc_id, article) pairs.
    Strict pair metrics are handled by pair_recall_at_k when relevant_pairs or
    required_pairs are available.
    """
    articles = {str(a).strip().lower() for a in relevant_articles if str(a).strip()}
    if not articles:
        return 0.0
    relevant_set = set(_unique_preserve_order(relevant_ids))
    got = {
        article
        for doc_id, article in _unique_pairs(retrieved_pairs)[:k]
        if doc_id in relevant_set and article in articles
    }
    return len(got) / len(articles)


def pair_hit_at_k(retrieved_pairs: list[tuple[str, str]], relevant_pairs: list[list[str]] | list[tuple[str, str]], k: int) -> float:
    relevant = set(_unique_pairs(relevant_pairs))
    if not relevant:
        return 0.0
    retrieved = _unique_pairs(retrieved_pairs)[:k]
    # Document-level pair (doc_id, "__full_doc__") is satisfied by any chunk from the doc.
    retrieved_doc_ids = {d for d, _ in retrieved}
    for pair in relevant:
        if pair in retrieved:
            return 1.0
        if _is_doc_level_pair(pair) and pair[0] in retrieved_doc_ids:
            return 1.0
    return 0.0


def pair_recall_at_k(retrieved_pairs: list[tuple[str, str]], relevant_pairs: list[list[str]] | list[tuple[str, str]], k: int) -> float:
    relevant = set(_unique_pairs(relevant_pairs))
    if not relevant:
        return 0.0
    retrieved = set(_unique_pairs(retrieved_pairs)[:k])
    retrieved_doc_ids = {d for d, _ in retrieved}
    hit = 0
    for pair in relevant:
        if pair in retrieved:
            hit += 1
        elif _is_doc_level_pair(pair) and pair[0] in retrieved_doc_ids:
            hit += 1
    return hit / len(relevant)


def doc_only_hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    return hit_at_k(retrieved_ids, relevant_ids, k)


def case_law_hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    case_law_ids = [x for x in relevant_ids if str(x).strip().endswith("_AL")]
    if not case_law_ids:
        return 0.0
    return hit_at_k(retrieved_ids, case_law_ids, k)


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
    relevant_pairs: list[list[str]] | list[tuple[str, str]] | None = None,
    required_pairs: list[list[str]] | list[tuple[str, str]] | None = None,
) -> dict:
    metrics = {
        "hit_at_k": hit_at_k(retrieved_ids, relevant_ids, k),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }
    if relevant_articles:
        metrics["article_hit_at_k"] = article_hit_at_k(retrieved_pairs or [], relevant_ids, relevant_articles, k)
        metrics["article_recall_at_k"] = article_recall_at_k(retrieved_pairs or [], relevant_ids, relevant_articles, k)
    if relevant_pairs:
        metrics["pair_hit_at_k"] = pair_hit_at_k(retrieved_pairs or [], relevant_pairs, k)
        metrics["pair_recall_at_k"] = pair_recall_at_k(retrieved_pairs or [], relevant_pairs, k)
    if required_pairs:
        metrics["required_pair_recall_at_k"] = pair_recall_at_k(retrieved_pairs or [], required_pairs, k)
        metrics["required_pair_full_hit_at_k"] = 1.0 if metrics["required_pair_recall_at_k"] >= 1.0 else 0.0
    if relevant_ids and not relevant_articles and not relevant_pairs and not required_pairs:
        metrics["doc_only_hit_at_k"] = doc_only_hit_at_k(retrieved_ids, relevant_ids, k)
    if any(str(x).strip().endswith("_AL") for x in relevant_ids):
        metrics["case_law_hit_at_k"] = case_law_hit_at_k(retrieved_ids, relevant_ids, k)
    return metrics


def average_metrics(metrics: list[dict]) -> dict:
    """Average metrics only over applicable cases.

    Earlier versions treated a missing metric as 0, so ArticleHit/PairHit were
    diluted by doc-only and case-law cases. This function reports both the
    metric and a `<metric>__n` support count to make legal evaluation fairer.
    """
    if not metrics:
        return {"hit_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "reciprocal_rank": 0.0}
    keys = sorted({key for item in metrics for key in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in metrics if key in item]
        if not vals:
            continue
        out[key] = sum(vals) / len(vals)
        out[f"{key}__n"] = float(len(vals))
    return out
