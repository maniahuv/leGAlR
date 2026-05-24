import time
from collections import defaultdict
from src.indexing.chroma_store import get_store
from src.indexing.bm25_index import load_bm25_index
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.evaluation.metrics import evaluate_retrieval_case, average_metrics
from src.retrieval.graph import graph_search

def _doc_id(doc) -> str:
    return str((doc.metadata or {}).get("doc_id", ""))

def run_retrieval_benchmark(
    test_cases: list[dict],
    strategy: str = "hybrid",
    k: int = 5,
) -> dict:
    store = get_store()
    bm25 = load_bm25_index()
    
    graph_obj = None
    if strategy == "graph":
        try:
            from src.tools.retrieval_tools import _get_graph
            graph_obj = _get_graph()
        except Exception:
            pass

    case_results = []
    scenario_metrics = defaultdict(list)

    for case in test_cases:
        query = case["query"]
        scenario = case.get("scenario", "single-hop-semantic")
        relevant_ids = [str(x) for x in case.get("relevant_ids", [])]

        if strategy == "dense":
            docs = dense_search(store, query, k=k)
        elif strategy == "hybrid":
            docs = hybrid_search(store, bm25, query, k=k)
        elif strategy == "hybrid_rerank":
            # Tăng ứng viên thô để Reranker có không gian xếp hạng trước khi hybrid trích xuất ID văn bản độc nhất
            candidates = hybrid_search(store, bm25, query, k=k*3)
            docs = rerank(query, candidates, k=k)
        elif strategy == "graph":
            if graph_obj is not None:
                docs = graph_search(store, graph_obj, query, k=k, initial_k=5, max_hops=2)
            else:
                docs = hybrid_search(store, bm25, query, k=k)

        retrieved_ids = [_doc_id(doc) for doc in docs][:k]

        metrics = evaluate_retrieval_case(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        scenario_metrics[scenario].append(metrics)
        scenario_metrics["global_overall"].append(metrics)

        case_results.append(
            {
                "query": query,
                "scenario": scenario,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "metrics": metrics,
            }
        )

    summary_by_scenario = {}
    for sc, metric_list in scenario_metrics.items():
        summary_by_scenario[sc] = average_metrics(metric_list)

    return {
        "strategy": strategy,
        "k": k,
        "summary": summary_by_scenario["global_overall"],
        "breakdown": summary_by_scenario,
        "cases": case_results,
    }