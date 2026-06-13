from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config
from src.evaluation.benchmark import run_retrieval_benchmark


def _default_testset() -> str:
    if str(getattr(config.dataset, "source", "huggingface")) == "local_pdf":
        return str(ROOT_DIR / "data" / "evaluation" / "family_law_test_cases.json")
    return str(ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=_default_testset())
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategies", nargs="+", default=["dense", "hybrid", "hybrid_rerank", "graph", "auto"])
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "evaluation" / "benchmark_results.json"))
    parser.add_argument("--show-failures", action="store_true", help="Print failed cases with retrieved doc/article pairs.")
    args = parser.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    all_results = {}
    print(f"Loaded {len(test_cases)} test cases")
    for strategy in args.strategies:
        print("\n" + "-" * 50)
        print(f"Strategy: {strategy.upper()}")
        print("-" * 50)
        result = run_retrieval_benchmark(test_cases, strategy=strategy, k=args.k)
        all_results[strategy] = result
        s = result["summary"]
        print(f"Hit@{args.k}:       {s.get('hit_at_k', 0):.4f}")
        print(f"Precision@{args.k}: {s.get('precision_at_k', 0):.4f}")
        print(f"Recall@{args.k}:    {s.get('recall_at_k', 0):.4f}")
        print(f"MRR:               {s.get('reciprocal_rank', 0):.4f}")
        if "article_hit_at_k" in s:
            n = int(s.get('article_hit_at_k__n', 0))
            print(f"ArticleHit@{args.k}: {s.get('article_hit_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "article_recall_at_k" in s:
            n = int(s.get('article_recall_at_k__n', 0))
            print(f"ArticleRecall@{args.k}: {s.get('article_recall_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "pair_hit_at_k" in s:
            n = int(s.get('pair_hit_at_k__n', 0))
            print(f"PairHit@{args.k}:    {s.get('pair_hit_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "pair_recall_at_k" in s:
            n = int(s.get('pair_recall_at_k__n', 0))
            print(f"PairRecall@{args.k}: {s.get('pair_recall_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "required_pair_recall_at_k" in s:
            n = int(s.get('required_pair_recall_at_k__n', 0))
            print(f"ReqPairRecall@{args.k}: {s.get('required_pair_recall_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "required_pair_full_hit_at_k" in s:
            n = int(s.get('required_pair_full_hit_at_k__n', 0))
            print(f"FullSupport@{args.k}: {s.get('required_pair_full_hit_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "case_law_hit_at_k" in s:
            n = int(s.get('case_law_hit_at_k__n', 0))
            print(f"CaseLawHit@{args.k}: {s.get('case_law_hit_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if "doc_only_hit_at_k" in s:
            n = int(s.get('doc_only_hit_at_k__n', 0))
            print(f"DocOnlyHit@{args.k}: {s.get('doc_only_hit_at_k', 0):.4f}" + (f" (n={n})" if n else ""))
        if result.get("route_counts"):
            print(f"Routes: {result['route_counts']}")
        if "avg_latency_ms" in result:
            print(f"Avg latency/case: {result['avg_latency_ms']:.1f} ms")
        if args.show_failures:
            failures = [
                c for c in result.get("cases", [])
                if not c.get("hit") or c.get("required_pair_full_hit") is False
            ]
            print(f"Failures: {len(failures)}")
            for c in failures[:20]:
                print(f"  - {c.get('id')}: {c.get('query')}")
                print(f"    expected={c.get('relevant_ids')} articles={c.get('relevant_articles')} required_pairs={c.get('required_pairs')}")
                print(f"    got={c.get('retrieved_pairs')}")
        print("Breakdown:")
        for scenario, metrics in result["breakdown"].items():
            if scenario == "global_overall":
                continue
            line = f"  {scenario}: Hit={metrics.get('hit_at_k',0):.3f} | P={metrics.get('precision_at_k',0):.3f} | R={metrics.get('recall_at_k',0):.3f} | MRR={metrics.get('reciprocal_rank',0):.3f}"
            if "article_hit_at_k" in metrics:
                line += f" | ArticleHit={metrics.get('article_hit_at_k',0):.3f}"
            if "pair_hit_at_k" in metrics:
                line += f" | PairHit={metrics.get('pair_hit_at_k',0):.3f}"
            if "case_law_hit_at_k" in metrics:
                line += f" | CaseLawHit={metrics.get('case_law_hit_at_k',0):.3f}"
            if "required_pair_recall_at_k" in metrics:
                line += f" | ReqPairRecall={metrics.get('required_pair_recall_at_k',0):.3f}"
            print(line)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
