from typing import Any


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Hit@K = trong top-k có ít nhất 1 tài liệu đúng hay không.
    """
    retrieved_top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    return 1.0 if any(doc_id in relevant_set for doc_id in retrieved_top_k) else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Precision@K = số tài liệu đúng trong top-k / k.
    """
    retrieved_top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    if k == 0:
        return 0.0

    correct = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)

    return correct / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@K = số tài liệu đúng tìm được trong top-k / tổng số tài liệu đúng.
    """
    if not relevant_ids:
        return 0.0

    retrieved_top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    correct = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)

    return correct / len(relevant_set)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Reciprocal Rank = 1 / vị trí của tài liệu đúng đầu tiên.
    """
    relevant_set = set(relevant_ids)

    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(results: list[dict[str, Any]]) -> float:
    """
    MRR trung bình trên nhiều query.

    Mỗi item cần có:
    {
        "retrieved_ids": [...],
        "relevant_ids": [...]
    }
    """
    if not results:
        return 0.0

    scores = [
        reciprocal_rank(
            item["retrieved_ids"],
            item["relevant_ids"],
        )
        for item in results
    ]

    return sum(scores) / len(scores)


def evaluate_retrieval_case(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> dict:
    """
    Tính toàn bộ metric retrieval cho một query.
    """
    return {
        "hit_at_k": hit_at_k(retrieved_ids, relevant_ids, k),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }


def average_metrics(metrics: list[dict]) -> dict:
    """
    Trung bình các metric trên nhiều query.
    """
    if not metrics:
        return {}

    keys = metrics[0].keys()

    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in keys
    }