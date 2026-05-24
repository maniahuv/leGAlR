from src.indexing.chroma_store import get_store
from src.indexing.bm25_index import load_bm25_index

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank

from src.evaluation.metrics import (
    evaluate_retrieval_case,
    average_metrics,
)


def _doc_id(doc) -> str:
    return str((doc.metadata or {}).get("doc_id", ""))


def run_retrieval_benchmark(
    test_cases: list[dict],
    strategy: str = "hybrid",
    k: int = 5,
) -> dict:
    """
    Chạy benchmark retrieval.

    test_cases format:
    [
        {
            "query": "...",
            "relevant_ids": ["123", "456"]
        }
    ]

    strategy:
    - dense
    - hybrid
    - hybrid_rerank
    """
    store = get_store()
    bm25 = load_bm25_index()

    case_results = []

    for case in test_cases:
        query = case["query"]
        relevant_ids = [str(x) for x in case.get("relevant_ids", [])]

        if strategy == "dense":
            docs = dense_search(store, query, k=k)

        elif strategy == "hybrid":
            docs = hybrid_search(store, bm25, query, k=k)

        elif strategy == "hybrid_rerank":
            candidates = hybrid_search(store, bm25, query, k=k * 2)
            docs = rerank(query, candidates, k=k)

        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        docs = _unique_docs_by_doc_id(docs)
        retrieved_ids = [_doc_id(doc) for doc in docs]

        metrics = evaluate_retrieval_case(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        case_results.append(
            {
                "query": query,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "retrieved_docs": [
                    {
                        "doc_id": str((doc.metadata or {}).get("doc_id", "")),
                        "title": (doc.metadata or {}).get("title", ""),
                        "so_ky_hieu": (doc.metadata or {}).get("so_ky_hieu", ""),
                        "tinh_trang_hieu_luc": (doc.metadata or {}).get("tinh_trang_hieu_luc", ""),
                        "preview": doc.page_content[:300],
                    }
                    for doc in docs
                ],
                "metrics": metrics,
            }
        )

    summary = average_metrics(
        [item["metrics"] for item in case_results]
    )

    return {
        "strategy": strategy,
        "k": k,
        "summary": summary,
        "cases": case_results,
    }

def _unique_docs_by_doc_id(docs):
    seen = set()
    result = []

    for doc in docs:
        doc_id = str((doc.metadata or {}).get("doc_id", ""))

        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            result.append(doc)

    return result