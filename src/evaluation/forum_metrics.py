from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def norm_doc_id(value: Any) -> str:
    return str(value or "").strip()


def norm_article(value: Any) -> str:
    """Normalize Vietnamese article labels.

    Examples:
    - "Điều 33" -> "33"
    - "33" -> "33"
    - "__full_doc__" -> "__full_doc__"
    """
    s = str(value or "").strip().lower()
    if not s:
        return ""
    if s in {"*", "__full_doc__", "full_doc", "doc", "document"}:
        return "__full_doc__"
    s = re.sub(r"^điều\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^dieu\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def is_doc_level_article(article: Any) -> bool:
    return norm_article(article) in {"", "__full_doc__"}


def is_case_law_doc(doc_id: str) -> bool:
    did = norm_doc_id(doc_id)
    return bool(re.search(r"_AL$|_AL_|^[0-9]+_20[0-9]{2}_AL", did))


def unique_preserve_order(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = norm_doc_id(item)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize_pair(raw: Any) -> tuple[str, str]:
    """Support both old pairs and dict pairs produced by forum benchmark.

    Supported inputs:
    - ["52_2014_QH13", "81"]
    - ("52_2014_QH13", "Điều 81")
    - {"doc_id": "52_2014_QH13", "article_number": "81"}
    - {"doc_id_full": "...", "article": "Điều 81"}
    """
    if isinstance(raw, dict):
        doc_id = raw.get("doc_id") or raw.get("doc") or raw.get("document_id") or raw.get("doc_id_full") or raw.get("document")
        article = raw.get("article_number") or raw.get("article") or raw.get("article_id") or raw.get("dieu")
        return (norm_doc_id(doc_id), norm_article(article))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (norm_doc_id(raw[0]), norm_article(raw[1]))
    return ("", "")


def unique_pairs(pairs: Iterable[Any]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for raw in pairs or []:
        pair = normalize_pair(raw)
        if pair[0] and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def case_expected_doc_ids(case: dict[str, Any]) -> list[str]:
    ids: list[Any] = []
    for key in ["relevant_ids", "expected_docs", "expected_doc_ids"]:
        ids.extend(_as_list(case.get(key)))

    # Keep full ids available for audit, but short ids usually drive retrieval.
    # If a benchmark accidentally only has full ids, these will still be checked.
    ids.extend(_as_list(case.get("relevant_doc_ids_full")))

    # required_pairs may contain doc_id even if relevant_ids is missing.
    for pair in unique_pairs(case.get("required_pairs") or []):
        ids.append(pair[0])

    return unique_preserve_order(ids)


def case_expected_articles(case: dict[str, Any]) -> list[str]:
    values = _as_list(case.get("relevant_articles") or case.get("expected_articles"))
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        article = norm_article(value)
        if article and article not in seen:
            seen.add(article)
            out.append(article)
    return out


def case_required_pairs(case: dict[str, Any]) -> list[tuple[str, str]]:
    return unique_pairs(case.get("required_pairs") or [])


def top_k_doc_ids(retrieved_ids: list[str], k: int) -> list[str]:
    return unique_preserve_order(retrieved_ids)[:k]


def top_k_pairs(retrieved_pairs: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
    return unique_pairs(retrieved_pairs)[:k]


def doc_hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    got = set(top_k_doc_ids(retrieved_ids, k))
    exp = set(unique_preserve_order(expected_ids))
    if not exp:
        return 0.0
    return 1.0 if got & exp else 0.0


def doc_precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    got = top_k_doc_ids(retrieved_ids, k)
    exp = set(unique_preserve_order(expected_ids))
    return sum(1 for x in got if x in exp) / k


def doc_recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    got = set(top_k_doc_ids(retrieved_ids, k))
    exp = set(unique_preserve_order(expected_ids))
    if not exp:
        return 0.0
    return len(got & exp) / len(exp)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    exp = set(unique_preserve_order(expected_ids))
    if not exp:
        return 0.0
    for i, did in enumerate(unique_preserve_order(retrieved_ids), start=1):
        if did in exp:
            return 1.0 / i
    return 0.0


def pair_satisfied(pair: tuple[str, str], retrieved_pairs: list[tuple[str, str]]) -> bool:
    doc_id, article = pair
    retrieved_docs = {d for d, _ in retrieved_pairs}
    if is_doc_level_article(article):
        return doc_id in retrieved_docs
    return pair in set(retrieved_pairs)


def pair_hit_at_k(retrieved_pairs: list[tuple[str, str]], expected_pairs: list[tuple[str, str]], k: int) -> float:
    exp = unique_pairs(expected_pairs)
    if not exp:
        return 0.0
    got = top_k_pairs(retrieved_pairs, k)
    return 1.0 if any(pair_satisfied(pair, got) for pair in exp) else 0.0


def pair_recall_at_k(retrieved_pairs: list[tuple[str, str]], expected_pairs: list[tuple[str, str]], k: int) -> float:
    exp = unique_pairs(expected_pairs)
    if not exp:
        return 0.0
    got = top_k_pairs(retrieved_pairs, k)
    satisfied = sum(1 for pair in exp if pair_satisfied(pair, got))
    return satisfied / len(exp)


def loose_article_hit_at_k(
    retrieved_pairs: list[tuple[str, str]],
    expected_doc_ids: list[str],
    expected_articles: list[str],
    k: int,
) -> float:
    docs = set(unique_preserve_order(expected_doc_ids))
    articles = {norm_article(a) for a in expected_articles if norm_article(a)}
    if not docs or not articles:
        return 0.0
    for did, article in top_k_pairs(retrieved_pairs, k):
        if did in docs and article in articles:
            return 1.0
    return 0.0


def loose_article_recall_at_k(
    retrieved_pairs: list[tuple[str, str]],
    expected_doc_ids: list[str],
    expected_articles: list[str],
    k: int,
) -> float:
    docs = set(unique_preserve_order(expected_doc_ids))
    articles = {norm_article(a) for a in expected_articles if norm_article(a)}
    if not docs or not articles:
        return 0.0
    got = {
        article
        for did, article in top_k_pairs(retrieved_pairs, k)
        if did in docs and article in articles
    }
    return len(got) / len(articles)


def evaluate_forum_case(
    retrieved_ids: list[str],
    retrieved_pairs: list[tuple[str, str]],
    case: dict[str, Any],
    k: int,
) -> dict[str, float]:
    expected_ids = case_expected_doc_ids(case)
    expected_articles = case_expected_articles(case)
    required_pairs = case_required_pairs(case)

    metrics: dict[str, float] = {
        "hit_at_k": doc_hit_at_k(retrieved_ids, expected_ids, k),
        "precision_at_k": doc_precision_at_k(retrieved_ids, expected_ids, k),
        "recall_at_k": doc_recall_at_k(retrieved_ids, expected_ids, k),
        "mrr": reciprocal_rank(retrieved_ids, expected_ids),
    }

    if expected_articles:
        metrics["article_hit_at_k"] = loose_article_hit_at_k(retrieved_pairs, expected_ids, expected_articles, k)
        metrics["article_recall_at_k"] = loose_article_recall_at_k(retrieved_pairs, expected_ids, expected_articles, k)

    if required_pairs:
        metrics["required_pair_hit_at_k"] = pair_hit_at_k(retrieved_pairs, required_pairs, k)
        metrics["required_pair_recall_at_k"] = pair_recall_at_k(retrieved_pairs, required_pairs, k)
        metrics["required_pair_full_hit_at_k"] = 1.0 if metrics["required_pair_recall_at_k"] >= 1.0 else 0.0

    if case.get("legal_domain") == "case_law" or any(is_case_law_doc(x) for x in expected_ids):
        case_ids = [x for x in expected_ids if is_case_law_doc(x)] or expected_ids
        metrics["case_law_hit_at_k"] = doc_hit_at_k(retrieved_ids, case_ids, k)

    return metrics


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in items if key in item]
        if not vals:
            continue
        out[key] = sum(vals) / len(vals)
        out[f"{key}__n"] = float(len(vals))
    return out


def breakdown_by(cases: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case in cases:
        label = str(case.get(group_key) or "unknown")
        buckets[label].append(case.get("metrics", {}))
    return {label: average_metric_dicts(metrics) for label, metrics in sorted(buckets.items())}
