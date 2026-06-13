from __future__ import annotations

from collections import Counter, defaultdict
import time

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
    meta = doc.metadata or {}
    did = str(meta.get("doc_id", "")).strip()
    art = str(meta.get("article", "")).strip()
    doc_type = str(meta.get("doc_type", "") or meta.get("corpus_role", "") or meta.get("source_class", "")).strip().lower()
    if not art and (did.endswith("_AL") or doc_type == "case_law"):
        return "__full_doc__"
    return art


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
    """Route used only for benchmark reporting.

    Production auto uses graph for relation/document-link questions and
    hybrid_rerank for normal legal QA. Dense remains a baseline only.
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
        relevant_pairs = case.get("relevant_pairs") or []
        if not relevant_pairs and relevant_articles:
            # Backward-compatible strict pairs for single-document cases.
            if len(relevant_ids) == 1:
                relevant_pairs = [[relevant_ids[0], article] for article in relevant_articles]
        required_pairs = case.get("required_pairs") or []

        t_case = time.perf_counter()

        if strategy == "dense":
            route = "dense"
            docs = dense_search(store, query, k=k)

        elif strategy in {"hybrid", "hybrid_rrf"}:
            route = "hybrid"
            docs = hybrid_search(store, bm25, query, k=k)

        elif strategy == "hybrid_rerank":
            route = "hybrid_rerank"
            # Use the production retrieval path so benchmark includes authority
            # injection and full-article context assembly.
            docs = retrieve_documents(query, k=k, strategy="hybrid_rerank")

        elif strategy == "graph":
            route = "graph"
            if graph_obj is not None:
                docs = graph_search(store, bm25, graph_obj, query, k=k, max_hops=1)
            else:
                docs = hybrid_search(store, bm25, query, k=k)

        elif strategy == "auto":
            # Benchmark must use the real production router in retrieval_tools.
            # Count the actual route written to metadata instead of a predicted
            # route label; otherwise route_counts can hide regressions.
            docs = retrieve_documents(query, k=k, strategy="auto")
            route = str((docs[0].metadata or {}).get("route", "auto")) if docs else "auto"

        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        latency_ms = (time.perf_counter() - t_case) * 1000

        route_counts[route] += 1
        retrieved_ids = _unique_doc_ids(docs, k)
        retrieved_pairs = _doc_article_pairs(docs, k)
        metrics = evaluate_retrieval_case(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
            retrieved_pairs=retrieved_pairs,
            relevant_articles=relevant_articles,
            relevant_pairs=relevant_pairs,
            required_pairs=required_pairs,
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
                "relevant_pairs": relevant_pairs,
                "required_pairs": required_pairs,
                "retrieved_ids": retrieved_ids,
                "retrieved_pairs": retrieved_pairs,
                "hit": bool(metrics.get("hit_at_k", 0.0)),
                "article_hit": bool(metrics.get("article_hit_at_k", 0.0)) if relevant_articles else None,
                "pair_hit": bool(metrics.get("pair_hit_at_k", 0.0)) if relevant_pairs else None,
                "case_law_hit": bool(metrics.get("case_law_hit_at_k", 0.0)) if any(str(x).endswith("_AL") for x in relevant_ids) else None,
                "required_pair_full_hit": bool(metrics.get("required_pair_full_hit_at_k", 0.0)) if required_pairs else None,
                "latency_ms": round(latency_ms, 3),
                "metrics": metrics,
            }
        )

    summary_by_scenario = {sc: average_metrics(metric_list) for sc, metric_list in scenario_metrics.items()}
    avg_latency_ms = sum(float(c.get("latency_ms", 0.0)) for c in case_results) / max(len(case_results), 1)
    return {
        "strategy": strategy,
        "k": k,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "summary": summary_by_scenario.get("global_overall", {}),
        "breakdown": summary_by_scenario,
        "route_counts": dict(route_counts),
        "cases": case_results,
    }
