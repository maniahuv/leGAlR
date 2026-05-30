from __future__ import annotations

from collections import defaultdict

from src.evaluation.metrics import average_metrics, evaluate_retrieval_case
from src.indexing.bm25_index import load_bm25_index
from src.indexing.chroma_store import get_store
from src.retrieval.dense import dense_search
from src.retrieval.graph import graph_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank


def _doc_id(doc) -> str:
    return str((doc.metadata or {}).get("doc_id", "")).strip()


def _unique_doc_ids(docs, k: int) -> list[str]:
    ids = []
    for doc in docs:
        did = _doc_id(doc)
        if did and did not in ids:
            ids.append(did)
        if len(ids) >= k:
            break
    return ids


def run_retrieval_benchmark(test_cases: list[dict], strategy: str = "hybrid", k: int = 5) -> dict:
    store = get_store()
    bm25 = load_bm25_index()

    graph_obj = None
    if strategy in {"graph", "auto"}:
        try:
            from src.tools.retrieval_tools import _get_graph
            graph_obj = _get_graph()
        except Exception as e:
            print(f"Could not load graph, fallback to hybrid. Error: {e}")

    case_results = []
    scenario_metrics = defaultdict(list)

    for case in test_cases:
        query = case["query"]
        scenario = case.get("scenario", "single-hop-semantic")
        relevant_ids = [str(x).strip() for x in case.get("relevant_ids", []) if str(x).strip()]

        if strategy == "dense":
            docs = dense_search(store, query, k=k)
        elif strategy in {"hybrid", "hybrid_rrf"}:
            docs = hybrid_search(store, bm25, query, k=k)
        elif strategy == "hybrid_rerank":
            candidates = hybrid_search(store, bm25, query, k=max(k * 8, 40))
            docs = rerank(query, candidates, k=k, force=True)
        elif strategy == "graph":
            docs = graph_search(store, bm25, graph_obj, query, k=k, max_hops=2) if graph_obj is not None else hybrid_search(store, bm25, query, k=k)
        elif strategy == "auto":
            if scenario == "multi-hop-graph" and graph_obj is not None:
                docs = graph_search(store, bm25, graph_obj, query, k=k, max_hops=2)
            else:
                candidates = hybrid_search(store, bm25, query, k=max(k * 6, 30))
                docs = rerank(query, candidates, k=k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        retrieved_ids = _unique_doc_ids(docs, k)
        metrics = evaluate_retrieval_case(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids, k=k)

        scenario_metrics[scenario].append(metrics)
        scenario_metrics["global_overall"].append(metrics)
        case_results.append({
            "id": case.get("id"),
            "query": query,
            "scenario": scenario,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics,
        })

    summary_by_scenario = {sc: average_metrics(metric_list) for sc, metric_list in scenario_metrics.items()}
    return {
        "strategy": strategy,
        "k": k,
        "summary": summary_by_scenario.get("global_overall", {}),
        "breakdown": summary_by_scenario,
        "cases": case_results,
    }
