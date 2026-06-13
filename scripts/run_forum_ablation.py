from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.forum_ablation import load_forum_benchmark, run_forum_ablation, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablation on the 300-question Danluat forum benchmark.")
    parser.add_argument(
        "--testset",
        default=str(ROOT_DIR / "data" / "evaluation" / "family_law_forum_benchmark_300.json"),
        help="Path to forum benchmark JSON/JSONL.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["dense", "bm25", "hybrid", "hybrid_rerank", "graph", "auto"],
        help="Strategies to evaluate: dense bm25 hybrid hybrid_rerank graph auto.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT_DIR / "data" / "evaluation" / "forum_ablation_300"),
        help="Output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit. 0 means all cases.",
    )
    args = parser.parse_args()

    cases = load_forum_benchmark(args.testset)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} forum benchmark cases")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"K={args.k}")

    results = {}
    for strategy in args.strategies:
        print("\n" + "-" * 60)
        print(f"Running strategy: {strategy.upper()}")
        print("-" * 60)
        one = run_forum_ablation(cases, strategies=[strategy], k=args.k)[strategy]
        results[strategy] = one
        s = one.get("summary", {})
        print(f"Hit@{args.k}:              {s.get('hit_at_k', 0):.4f}")
        print(f"Precision@{args.k}:        {s.get('precision_at_k', 0):.4f}")
        print(f"Recall@{args.k}:           {s.get('recall_at_k', 0):.4f}")
        print(f"MRR:                  {s.get('mrr', 0):.4f}")
        if "article_hit_at_k" in s:
            print(f"ArticleHit@{args.k}:       {s.get('article_hit_at_k', 0):.4f}")
        if "required_pair_recall_at_k" in s:
            print(f"ReqPairRecall@{args.k}:    {s.get('required_pair_recall_at_k', 0):.4f}")
        if "required_pair_full_hit_at_k" in s:
            print(f"FullSupport@{args.k}:      {s.get('required_pair_full_hit_at_k', 0):.4f}")
        if "case_law_hit_at_k" in s:
            print(f"CaseLawHit@{args.k}:       {s.get('case_law_hit_at_k', 0):.4f}")
        print(f"Avg latency/case:     {one.get('avg_latency_ms', 0):.1f} ms")
        if one.get("route_counts"):
            print(f"Routes:               {json.dumps(one['route_counts'], ensure_ascii=False)}")

    write_outputs(results, args.out_dir, k=args.k)
    print("\nSaved outputs to:")
    print(f"  {args.out_dir}")
    print("Main files:")
    print("  forum_ablation_report.md")
    print("  forum_ablation_summary.csv")
    print("  forum_ablation_failures.csv")
    print("  forum_ablation_results.json")


if __name__ == "__main__":
    main()
