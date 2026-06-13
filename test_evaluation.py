from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.benchmark import run_retrieval_benchmark


def main():
    test_cases_path = ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json"
    if not test_cases_path.exists():
        raise FileNotFoundError("Missing test cases. Run: python scripts/generate_test_cases.py")
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    strategies = ["dense", "hybrid", "hybrid_rerank", "graph", "auto"]
    print("=" * 90)
    print(f"🚀 CHẠY BENCHMARK RETRIEVAL ({len(test_cases)} CÂU HỎI)")
    print("=" * 90)
    for strategy in strategies:
        print("\n" + "-" * 40)
        print(f"Strategy: {strategy.upper()}")
        print("-" * 40)
        result = run_retrieval_benchmark(test_cases=test_cases, strategy=strategy, k=5)
        summary = result["summary"]
        print("\n[GLOBAL OVERALL METRICS]")
        print(f" -> Hit@5:       {summary['hit_at_k']:.4f}")
        print(f" -> Precision@5: {summary['precision_at_k']:.4f}")
        print(f" -> Recall@5:    {summary['recall_at_k']:.4f}")
        print(f" -> MRR:         {summary['reciprocal_rank']:.4f}")
        print("\n[BREAKDOWN BY SCENARIOS]")
        for scenario, metrics in result["breakdown"].items():
            if scenario == "global_overall":
                continue
            print(f" 📂 Phân đoạn: {scenario}")
            print(f"    * Hit@5: {metrics['hit_at_k']:.3f} | Precision@5: {metrics['precision_at_k']:.3f} | Recall@5: {metrics['recall_at_k']:.3f} | MRR: {metrics['reciprocal_rank']:.3f}")


if __name__ == "__main__":
    main()
