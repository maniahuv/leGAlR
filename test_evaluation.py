import sys
import json
from pathlib import Path
from pprint import pprint

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.benchmark import run_retrieval_benchmark


with open(ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

for strategy in ["dense", "hybrid", "hybrid_rerank"]:
    print("=" * 80)
    print("Strategy:", strategy)

    result = run_retrieval_benchmark(
        test_cases=test_cases,
        strategy=strategy,
        k=5,
    )

    pprint(result["summary"])