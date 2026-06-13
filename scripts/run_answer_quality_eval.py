from __future__ import annotations

"""Generate reviewable answers for manual answer-quality evaluation.

Example:
python scripts/run_answer_quality_eval.py ^
  --benchmark data/evaluation/family_law_forum_benchmark_300.json ^
  --limit 50 ^
  --strategy auto ^
  --k 5 ^
  --out-dir data/evaluation/answer_quality_eval
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.answer_quality import (  # noqa: E402
    load_benchmark,
    make_review_row,
    sample_cases,
    write_csv,
    write_jsonl,
    write_rubric,
)
from src.llm import get_llm  # noqa: E402
from src.tools.retrieval_tools import format_docs_for_context_with_query, retrieve_documents  # noqa: E402

try:  # Reuse API prompt/cleaning so evaluation matches production behavior.
    from src.api.routers.query import _build_answer_prompt, _clean_answer, _llm_output_to_text  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    def _build_answer_prompt(question: str, context: str, answer_style: str = "normal") -> str:
        return f"""
Bạn là trợ lý pháp lý hỗ trợ tra cứu pháp luật Việt Nam trong lĩnh vực Hôn nhân và Gia đình.
Chỉ trả lời dựa trên NGỮ CẢNH PHÁP LÝ, không bịa điều luật. Nếu ngữ cảnh chưa đủ, nói rõ.

NGỮ CẢNH PHÁP LÝ:
{context}

CÂU HỎI:
{question}

YÊU CẦU:
- Trả lời bằng tiếng Việt.
- Cấu trúc: Kết luận, Căn cứ, Áp dụng, Lưu ý nếu cần.
TRẢ LỜI:
""".strip()

    def _llm_output_to_text(output: Any) -> str:
        content = getattr(output, "content", output)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text") or item.get("content") or item) if isinstance(item, dict) else str(item) for item in content)
        return str(content or "")

    def _clean_answer(answer: str, question: str) -> str:
        return (answer or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer-quality sample generation for manual review.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON/JSONL/CSV.")
    parser.add_argument("--out-dir", default="data/evaluation/answer_quality_eval", help="Output directory.")
    parser.add_argument("--limit", type=int, default=50, help="Number of cases to generate. <=0 means all cases.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-mode", choices=["stratified", "random", "first"], default="stratified")
    parser.add_argument("--strategy", default="auto", choices=["auto", "dense", "bm25", "hybrid", "hybrid_rerank", "graph", "authority_dense", "authority_hybrid"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answer-style", choices=["short", "normal", "detailed"], default="normal")
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--no-llm", action="store_true", help="Only retrieve sources; leave answer blank. Useful to test pipeline/API keys.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls to avoid rate limits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_benchmark(args.benchmark)
    selected = sample_cases(cases, limit=args.limit, seed=args.seed, mode=args.sample_mode)

    print(f"Loaded {len(cases)} benchmark cases")
    print(f"Selected {len(selected)} cases for answer-quality evaluation")
    print(f"Strategy={args.strategy}, k={args.k}, answer_style={args.answer_style}")

    llm = None if args.no_llm else get_llm()
    rows: list[dict[str, Any]] = []

    for idx, case in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {case.id}: {case.query[:90]}")
        timings: dict[str, Any] = {}
        error = ""
        answer = ""
        docs = []
        t_case = time.perf_counter()

        try:
            t0 = time.perf_counter()
            docs = retrieve_documents(case.query, k=args.k, strategy=args.strategy)
            timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            t0 = time.perf_counter()
            context = format_docs_for_context_with_query(docs, case.query)[: args.max_context_chars]
            timings["context_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            if not args.no_llm:
                t0 = time.perf_counter()
                prompt = _build_answer_prompt(case.query, context, args.answer_style)
                raw = llm.invoke(prompt)
                answer = _clean_answer(_llm_output_to_text(raw), case.query)
                timings["llm_generation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")

        timings["total_ms"] = round((time.perf_counter() - t_case) * 1000, 2)
        rows.append(make_review_row(case=case, docs=docs, answer=answer, timings=timings, error=error))

        if args.sleep > 0 and idx < len(selected):
            time.sleep(args.sleep)

    csv_path = out_dir / "answer_quality_sample.csv"
    jsonl_path = out_dir / "answer_quality_sample.jsonl"
    rubric_path = out_dir / "answer_quality_rubric.md"

    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)
    write_rubric(rubric_path)

    print("\nSaved outputs to:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    print(f"  {rubric_path}")
    print("\nManual scoring columns use 0=sai, 1=đạt một phần, 2=đạt tốt.")


if __name__ == "__main__":
    main()
