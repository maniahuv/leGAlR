from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.evaluation.forum_metrics import average_metric_dicts, breakdown_by, evaluate_forum_case, norm_article
from src.indexing.bm25_index import load_bm25_index
from src.indexing.chroma_store import get_store
from src.retrieval.dense import dense_search
from src.retrieval.graph import graph_search
from src.retrieval.hybrid import hybrid_search
from src.tools.retrieval_tools import retrieve_documents


DEFAULT_GROUP_KEYS = ["legal_domain", "legal_issue", "technical_challenge", "scope", "difficulty", "scenario"]


def load_forum_benchmark(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError(f"Benchmark must be a list or {{'cases': [...]}}: {path}")
    return data


def _doc_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("doc_id", "")).strip()


def _article(doc: Document) -> str:
    meta = doc.metadata or {}
    did = str(meta.get("doc_id", "") or "").strip()
    article = str(meta.get("article", "") or "").strip()
    doc_type = str(meta.get("doc_type", "") or meta.get("corpus_role", "") or meta.get("source_class", "") or "").lower()
    if not article and ("_AL" in did or doc_type == "case_law"):
        return "__full_doc__"
    return norm_article(article)


def _unique_doc_ids(docs: list[Document], k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        did = _doc_id(doc)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
        if len(out) >= k:
            break
    return out


def _doc_article_pairs(docs: list[Document], k: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        pair = (_doc_id(doc), _article(doc))
        if pair[0] and pair not in seen:
            seen.add(pair)
            out.append(pair)
        if len(out) >= k:
            break
    return out


def _bm25_search(bm25, query: str, k: int) -> list[Document]:
    old_k = getattr(bm25, "k", None)
    try:
        setattr(bm25, "k", k)
        return bm25.invoke(query)
    finally:
        if old_k is not None:
            setattr(bm25, "k", old_k)


def _load_graph_for_benchmark():
    try:
        from src.tools.retrieval_tools import _get_graph

        return _get_graph()
    except Exception as exc:
        print(f"[WARN] Could not load graph. Graph strategy will fallback to hybrid. Error: {exc}")
        return None


def retrieve_by_strategy(strategy: str, query: str, k: int, store, bm25, graph_obj=None) -> tuple[list[Document], str]:
    strategy = strategy.lower().strip()

    if strategy == "dense":
        return dense_search(store, query, k=k), "dense"

    if strategy == "bm25":
        return _bm25_search(bm25, query, k=k), "bm25"

    if strategy in {"hybrid", "hybrid_rrf"}:
        return hybrid_search(store, bm25, query, k=k), "hybrid"

    if strategy in {"hybrid_rerank", "rerank"}:
        docs = retrieve_documents(query, k=k, strategy="hybrid_rerank")
        route = str((docs[0].metadata or {}).get("route", "hybrid_rerank")) if docs else "hybrid_rerank"
        return docs, route

    if strategy == "graph":
        if graph_obj is not None:
            return graph_search(store, bm25, graph_obj, query, k=k, max_hops=1), "graph"
        return hybrid_search(store, bm25, query, k=k), "graph_fallback_hybrid"

    if strategy == "auto":
        docs = retrieve_documents(query, k=k, strategy="auto")
        route = str((docs[0].metadata or {}).get("route", "auto")) if docs else "auto"
        return docs, route

    raise ValueError(f"Unknown strategy: {strategy}")


def run_forum_strategy(cases: list[dict[str, Any]], strategy: str, k: int = 5) -> dict[str, Any]:
    store = get_store()
    bm25 = load_bm25_index()
    graph_obj = _load_graph_for_benchmark() if strategy.lower() == "graph" else None

    case_results: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()

    for i, case in enumerate(cases, start=1):
        query = str(case.get("query") or case.get("question") or case.get("benchmark_query") or case.get("title") or "").strip()
        if not query:
            raise ValueError(f"Case #{i} has no query/title: {case.get('id')}")

        t0 = time.perf_counter()
        docs, route = retrieve_by_strategy(strategy, query, k, store, bm25, graph_obj)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_ids = _unique_doc_ids(docs, k)
        retrieved_pairs = _doc_article_pairs(docs, k)
        metrics = evaluate_forum_case(retrieved_ids, retrieved_pairs, case, k)
        route_counts[route] += 1

        case_result = {
            "id": case.get("id"),
            "query": query,
            "strategy": strategy,
            "route": route,
            "latency_ms": round(latency_ms, 3),
            "retrieved_ids": retrieved_ids,
            "retrieved_pairs": retrieved_pairs,
            "metrics": metrics,
            "hit": bool(metrics.get("hit_at_k", 0.0)),
            "required_pair_full_hit": bool(metrics.get("required_pair_full_hit_at_k", 0.0)) if "required_pair_full_hit_at_k" in metrics else None,
            "required_pair_recall": metrics.get("required_pair_recall_at_k"),
        }

        # Preserve useful labels for breakdown.
        for key in ["legal_domain", "legal_issue", "technical_challenge", "scope", "difficulty", "scenario", "url", "title"]:
            if key in case:
                case_result[key] = case.get(key)

        case_results.append(case_result)

    summary = average_metric_dicts([c["metrics"] for c in case_results])
    avg_latency = sum(float(c["latency_ms"]) for c in case_results) / max(len(case_results), 1)

    return {
        "strategy": strategy,
        "k": k,
        "num_cases": len(cases),
        "avg_latency_ms": round(avg_latency, 3),
        "route_counts": dict(route_counts),
        "summary": summary,
        "breakdown": {key: breakdown_by(case_results, key) for key in DEFAULT_GROUP_KEYS},
        "cases": case_results,
    }


def run_forum_ablation(cases: list[dict[str, Any]], strategies: list[str], k: int = 5) -> dict[str, Any]:
    return {strategy: run_forum_strategy(cases, strategy=strategy, k=k) for strategy in strategies}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_summary_csv(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "strategy",
        "num_cases",
        "hit_at_k",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "article_hit_at_k",
        "article_recall_at_k",
        "required_pair_hit_at_k",
        "required_pair_recall_at_k",
        "required_pair_full_hit_at_k",
        "case_law_hit_at_k",
        "avg_latency_ms",
        "route_counts",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for strategy, result in results.items():
            s = result.get("summary", {})
            writer.writerow({
                "strategy": strategy,
                "num_cases": result.get("num_cases"),
                "hit_at_k": s.get("hit_at_k"),
                "precision_at_k": s.get("precision_at_k"),
                "recall_at_k": s.get("recall_at_k"),
                "mrr": s.get("mrr"),
                "article_hit_at_k": s.get("article_hit_at_k"),
                "article_recall_at_k": s.get("article_recall_at_k"),
                "required_pair_hit_at_k": s.get("required_pair_hit_at_k"),
                "required_pair_recall_at_k": s.get("required_pair_recall_at_k"),
                "required_pair_full_hit_at_k": s.get("required_pair_full_hit_at_k"),
                "case_law_hit_at_k": s.get("case_law_hit_at_k"),
                "avg_latency_ms": result.get("avg_latency_ms"),
                "route_counts": json.dumps(result.get("route_counts", {}), ensure_ascii=False),
            })


def write_breakdown_csv(results: dict[str, Any], group_key: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for strategy, result in results.items():
        groups = result.get("breakdown", {}).get(group_key, {})
        for label, metrics in groups.items():
            rows.append({
                "strategy": strategy,
                group_key: label,
                "n": int(metrics.get("hit_at_k__n", 0)),
                "hit_at_k": metrics.get("hit_at_k"),
                "precision_at_k": metrics.get("precision_at_k"),
                "recall_at_k": metrics.get("recall_at_k"),
                "mrr": metrics.get("mrr"),
                "article_hit_at_k": metrics.get("article_hit_at_k"),
                "required_pair_recall_at_k": metrics.get("required_pair_recall_at_k"),
                "required_pair_full_hit_at_k": metrics.get("required_pair_full_hit_at_k"),
                "case_law_hit_at_k": metrics.get("case_law_hit_at_k"),
            })

    columns = ["strategy", group_key, "n", "hit_at_k", "precision_at_k", "recall_at_k", "mrr", "article_hit_at_k", "required_pair_recall_at_k", "required_pair_full_hit_at_k", "case_law_hit_at_k"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_failures_csv(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "strategy", "id", "query", "legal_domain", "legal_issue", "technical_challenge", "difficulty",
        "hit", "required_pair_recall", "required_pair_full_hit", "retrieved_ids", "retrieved_pairs", "url",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for strategy, result in results.items():
            for c in result.get("cases", []):
                is_failure = not c.get("hit") or c.get("required_pair_full_hit") is False
                if not is_failure:
                    continue
                writer.writerow({
                    "strategy": strategy,
                    "id": c.get("id"),
                    "query": c.get("query"),
                    "legal_domain": c.get("legal_domain"),
                    "legal_issue": c.get("legal_issue"),
                    "technical_challenge": c.get("technical_challenge"),
                    "difficulty": c.get("difficulty"),
                    "hit": c.get("hit"),
                    "required_pair_recall": c.get("required_pair_recall"),
                    "required_pair_full_hit": c.get("required_pair_full_hit"),
                    "retrieved_ids": json.dumps(c.get("retrieved_ids", []), ensure_ascii=False),
                    "retrieved_pairs": json.dumps(c.get("retrieved_pairs", []), ensure_ascii=False),
                    "url": c.get("url"),
                })


def write_markdown_report(results: dict[str, Any], path: str | Path, k: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Forum QA Ablation Report @ {k}")
    lines.append("")
    lines.append("## Global summary")
    lines.append("")
    lines.append("| Strategy | Hit | Recall | MRR | ArticleHit | ReqPairRecall | FullSupport | CaseLawHit | Latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for strategy, result in results.items():
        s = result.get("summary", {})
        lines.append(
            f"| {strategy} | {_fmt(s.get('hit_at_k'))} | {_fmt(s.get('recall_at_k'))} | {_fmt(s.get('mrr'))} | "
            f"{_fmt(s.get('article_hit_at_k'))} | {_fmt(s.get('required_pair_recall_at_k'))} | "
            f"{_fmt(s.get('required_pair_full_hit_at_k'))} | {_fmt(s.get('case_law_hit_at_k'))} | {_fmt(result.get('avg_latency_ms'), 1)} |"
        )

    for group_key in ["legal_domain", "technical_challenge", "difficulty"]:
        lines.append("")
        lines.append(f"## Breakdown by `{group_key}`")
        lines.append("")
        lines.append("| Strategy | Group | n | Hit | Recall | MRR | ReqPairRecall | CaseLawHit |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for strategy, result in results.items():
            groups = result.get("breakdown", {}).get(group_key, {})
            for label, metrics in groups.items():
                lines.append(
                    f"| {strategy} | {label} | {int(metrics.get('hit_at_k__n', 0))} | {_fmt(metrics.get('hit_at_k'))} | "
                    f"{_fmt(metrics.get('recall_at_k'))} | {_fmt(metrics.get('mrr'))} | "
                    f"{_fmt(metrics.get('required_pair_recall_at_k'))} | {_fmt(metrics.get('case_law_hit_at_k'))} |"
                )

    lines.append("")
    lines.append("## Reading notes")
    lines.append("")
    lines.append("- `Hit` checks whether at least one expected document appears in Top-K.")
    lines.append("- `ReqPairRecall` checks strict `(document, article)` support pairs where the benchmark provides them.")
    lines.append("- `FullSupport` is strict: all required pairs must appear in Top-K, so it can be low when one question needs many legal articles.")
    lines.append("- Forum benchmark is naturally skewed toward procedural and civil-status questions, so use it as a real-user benchmark rather than a balanced synthetic test.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: dict[str, Any], out_dir: str | Path, k: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forum_ablation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(results, out_dir / "forum_ablation_summary.csv")
    for group_key in DEFAULT_GROUP_KEYS:
        write_breakdown_csv(results, group_key, out_dir / f"forum_ablation_by_{group_key}.csv")
    write_failures_csv(results, out_dir / "forum_ablation_failures.csv")
    write_markdown_report(results, out_dir / "forum_ablation_report.md", k=k)
