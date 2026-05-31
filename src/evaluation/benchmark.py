from __future__ import annotations

from collections import Counter, defaultdict

from src.evaluation.metrics import average_metrics, evaluate_retrieval_case
from src.indexing.bm25_index import load_bm25_index
from src.indexing.chroma_store import get_store
from src.retrieval.dense import dense_search
from src.retrieval.graph import graph_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.tools.retrieval_tools import is_relation_query, retrieve_documents


def _doc_id(doc) -> str:
    return str((doc.metadata or {}).get("doc_id", "")).strip()


def _article(doc) -> str:
    return str((doc.metadata or {}).get("article", "")).strip()


def _unique_doc_ids(docs, k: int) -> list[str]:
    ids: list[str] = []
    for doc in docs:
        did = _doc_id(doc)
        if did and did not in ids:
            ids.append(did)
        if len(ids) >= k:
            break
    return ids


def _doc_article_pairs(docs, k: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        pair = (_doc_id(doc), _article(doc))
        if pair[0] and pair not in seen:
            pairs.append(pair)
            seen.add(pair)
        if len(pairs) >= k:
            break
    return pairs


def _pool_size(k: int, accurate: bool = False) -> int:
    from configs.setting import config

    multiplier = int(getattr(config.retrieval, "pool_multiplier", 6))
    minimum = int(getattr(config.retrieval, "min_pool_size", 30))
    if accurate:
        multiplier = max(multiplier, int(getattr(config.retrieval, "accurate_pool_multiplier", 8)))
        minimum = max(minimum, int(getattr(config.retrieval, "accurate_min_pool_size", 40)))
    return max(k * multiplier, minimum)


def _auto_route(query: str) -> str:
    """
    Route used only for benchmark reporting.

    The actual retrieval is performed by retrieve_documents(strategy="auto").
    This helper keeps route_counts consistent with the same relation-only router
    used in src.tools.retrieval_tools, and avoids using the old route_query()
    from legal_signals.
    """
    return "graph" if is_relation_query(query) else "hybrid_rerank"


def run_retrieval_benchmark(test_cases: list[dict], strategy: str = "hybrid", k: int = 5) -> dict:
    store = get_store()
    bm25 = load_bm25_index()

    graph_obj = None
    if strategy == "graph":
        try:
            from src.tools.retrieval_tools import _get_graph

            graph_obj = _get_graph()
        except Exception as e:
            print(f"Could not load graph, fallback to hybrid. Error: {e}")

    case_results: list[dict] = []
    scenario_metrics = defaultdict(list)
    route_counts: Counter[str] = Counter()

    for case in test_cases:
        query = case["query"]
        scenario = case.get("scenario", "single-hop-semantic")
        relevant_ids = [str(x).strip() for x in case.get("relevant_ids", []) if str(x).strip()]
        relevant_articles = [str(x).strip() for x in case.get("relevant_articles", []) if str(x).strip()]

        if strategy == "dense":
            route = "dense"
            docs = dense_search(store, query, k=k)

        elif strategy in {"hybrid", "hybrid_rrf"}:
            route = "hybrid"
            docs = hybrid_search(store, bm25, query, k=k)

        elif strategy == "hybrid_rerank":
            route = "hybrid_rerank"
            candidates = hybrid_search(store, bm25, query, k=_pool_size(k, accurate=True))
            docs = rerank(query, candidates, k=k, force=True)

        elif strategy == "graph":
            route = "graph"
            if graph_obj is not None:
                docs = graph_search(store, bm25, graph_obj, query, k=k, max_hops=1)
            else:
                docs = hybrid_search(store, bm25, query, k=k)

        elif strategy == "auto":
            # IMPORTANT: do not call legal_signals.route_query() here.
            # Benchmark must use the real production router in retrieval_tools.
            route = _auto_route(query)
            docs = retrieve_documents(query, k=k, strategy="auto")

        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        route_counts[route] += 1
        retrieved_ids = _unique_doc_ids(docs, k)
        retrieved_pairs = _doc_article_pairs(docs, k)
        metrics = evaluate_retrieval_case(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
            retrieved_pairs=retrieved_pairs,
            relevant_articles=relevant_articles,
        )

        scenario_metrics[scenario].append(metrics)
        scenario_metrics["global_overall"].append(metrics)
        case_results.append(
            {
                "id": case.get("id"),
                "query": query,
                "scenario": scenario,
                "route": route,
                "relevant_ids": relevant_ids,
                "relevant_articles": relevant_articles,
                "retrieved_ids": retrieved_ids,
                "retrieved_pairs": retrieved_pairs,
                "metrics": metrics,
            }
        )

    summary_by_scenario = {sc: average_metrics(metric_list) for sc, metric_list in scenario_metrics.items()}
    return {
        "strategy": strategy,
        "k": k,
        "summary": summary_by_scenario.get("global_overall", {}),
        "breakdown": summary_by_scenario,
        "route_counts": dict(route_counts),
        "cases": case_results,
    }
