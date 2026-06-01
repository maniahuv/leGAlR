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
            print(f"ArticleHit@{args.k}: {s.get('article_hit_at_k', 0):.4f}")
        if result.get("route_counts"):
            print(f"Routes: {result['route_counts']}")
        print("Breakdown:")
        for scenario, metrics in result["breakdown"].items():
            if scenario == "global_overall":
                continue
            line = f"  {scenario}: Hit={metrics.get('hit_at_k',0):.3f} | P={metrics.get('precision_at_k',0):.3f} | R={metrics.get('recall_at_k',0):.3f} | MRR={metrics.get('reciprocal_rank',0):.3f}"
            if "article_hit_at_k" in metrics:
                line += f" | ArticleHit={metrics.get('article_hit_at_k',0):.3f}"
            print(line)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
