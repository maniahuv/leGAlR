from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.llm import get_llm
from src.tools.retrieval_tools import (
    format_docs_for_context,
    generate_answer_tool,
    retrieve_documents,
)


def llm_output_to_text(raw) -> str:
    """
    Chuyển output của LLM về string.
    Hỗ trợ các dạng:
    - str
    - list content blocks
    - dict
    - AIMessage hoặc object có thuộc tính .content
    """
    if raw is None:
        return ""

    if hasattr(raw, "content"):
        raw = raw.content

    if isinstance(raw, str):
        return raw

    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(raw, dict):
        if "text" in raw:
            return str(raw["text"])
        if "content" in raw:
            return str(raw["content"])
        return str(raw)

    return str(raw)


def clean_answer(answer, question: str) -> str:
    answer = llm_output_to_text(answer).strip()
    question = (question or "").strip()

    for marker in ["[TRẢ LỜI]:", "[TRẢ LỜI]", "Trả lời:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    # Nếu model chỉ lặp lại câu hỏi
    if answer.lower().strip(" ?.!") == question.lower().strip(" ?.!"):
        return "Lỗi: model chỉ lặp lại câu hỏi, chưa sinh được câu trả lời."

    # Nếu câu trả lời mở đầu bằng chính câu hỏi
    if answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip(" \n:.-")

    return answer


def fmt_time(seconds: float) -> str:
    return f"{seconds:.3f}s"


def print_stats(name: str, values: list[float]) -> None:
    if not values:
        return

    print(f"{name}:")
    print(f"  avg: {fmt_time(mean(values))}")
    print(f"  min: {fmt_time(min(values))}")
    print(f"  max: {fmt_time(max(values))}")


if __name__ == "__main__":
    question = "Tòa án giải quyết thuận tình ly hôn khi vợ chồng đáp ứng những điều kiện nào?"

    # Nên dùng dense để test tốc độ trước, vì benchmark hiện tại dense tốt hơn hybrid_rerank.
    # Nếu muốn so sánh thì đổi lại thành: "hybrid_rerank", "hybrid", "graph", "auto".
    strategy = "dense"
    k = 3

    # Số lần đo sau warm-up
    num_runs = 5

    # Cắt context để giảm prompt size
    max_context_chars = 3000

    # Tắt LLM để đo riêng retrieval + prompt, tránh lỗi quota Gemini.
    # Khi cần đo end-to-end thì đổi thành True.
    enable_llm = False

    print("\n================= QUESTION =================")
    print(question)
    print(f"Strategy: {strategy} | k={k}")
    print(f"Runs after warm-up: {num_runs}")
    print(f"LLM enabled: {enable_llm}")

    # =========================================================
    # 0. Warm-up: load model/vector store/cache trong cùng process
    # =========================================================
    print("\n================= WARM-UP =================")
    t0 = time.perf_counter()
    warmup_docs = retrieve_documents(question, k=k, strategy=strategy)
    warmup_context = format_docs_for_context(warmup_docs)[:max_context_chars]
    warmup_prompt = generate_answer_tool.invoke({
        "query": question,
        "context": warmup_context,
    })
    warmup_time = time.perf_counter() - t0
    print(f"Warm-up time: {fmt_time(warmup_time)}")

    # Load LLM một lần nếu cần đo end-to-end
    llm = None
    llm_load_time = 0.0

    if enable_llm:
        t0 = time.perf_counter()
        llm = get_llm()
        llm_load_time = time.perf_counter() - t0
        print(f"Get LLM time: {fmt_time(llm_load_time)}")

    # =========================================================
    # 1. Benchmark nhiều lần trong cùng process
    # =========================================================
    retrieval_times: list[float] = []
    format_times: list[float] = []
    prompt_times: list[float] = []
    llm_times: list[float] = []
    clean_times: list[float] = []
    total_times: list[float] = []

    last_docs = None
    last_context = ""
    last_answer = ""

    print("\n================= BENCHMARK RUNS =================")

    for i in range(num_runs):
        run_start = time.perf_counter()

        # 1. Retrieval
        t0 = time.perf_counter()
        docs = retrieve_documents(question, k=k, strategy=strategy)
        retrieval_time = time.perf_counter() - t0

        # 2. Format context
        t0 = time.perf_counter()
        context = format_docs_for_context(docs)[:max_context_chars]
        format_time = time.perf_counter() - t0

        # 3. Build prompt
        t0 = time.perf_counter()
        prompt = generate_answer_tool.invoke({
            "query": question,
            "context": context,
        })
        prompt_time = time.perf_counter() - t0

        # 4. Optional LLM generation
        llm_generation_time = 0.0
        clean_time = 0.0
        answer = "Đã bỏ qua bước gọi LLM để đo riêng retrieval/prompt."

        if enable_llm and llm is not None:
            try:
                t0 = time.perf_counter()
                response = llm.invoke(prompt)
                llm_generation_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                answer = clean_answer(response, question)
                clean_time = time.perf_counter() - t0

            except Exception as e:
                llm_generation_time = time.perf_counter() - t0
                clean_time = 0.0
                answer = f"Lỗi khi gọi LLM: {type(e).__name__}: {e}"

        total_time = time.perf_counter() - run_start

        retrieval_times.append(retrieval_time)
        format_times.append(format_time)
        prompt_times.append(prompt_time)
        llm_times.append(llm_generation_time)
        clean_times.append(clean_time)
        total_times.append(total_time)

        last_docs = docs
        last_context = context
        last_answer = answer

        print(
            f"Run {i + 1}: "
            f"retrieval={fmt_time(retrieval_time)} | "
            f"format={fmt_time(format_time)} | "
            f"prompt={fmt_time(prompt_time)} | "
            f"llm={fmt_time(llm_generation_time)} | "
            f"total={fmt_time(total_time)}"
        )

    # =========================================================
    # 2. In context của lần cuối để kiểm tra chất lượng retrieval
    # =========================================================
    print("\n================= RETRIEVED CONTEXT - LAST RUN =================")
    print(last_context[:4000])

    if enable_llm:
        print("\n================= ANSWER - LAST RUN =================")
        print(last_answer)

    # =========================================================
    # 3. Tổng hợp timing
    # =========================================================
    print("\n================= TIMING SUMMARY =================")
    print(f"Warm-up time: {fmt_time(warmup_time)}")
    if enable_llm:
        print(f"Get LLM time: {fmt_time(llm_load_time)}")
    print("------------------------------------------")
    print_stats("Retrieval time", retrieval_times)
    print_stats("Format context time", format_times)
    print_stats("Build prompt time", prompt_times)

    if enable_llm:
        print_stats("LLM generation time", llm_times)
        print_stats("Clean answer time", clean_times)

    print_stats("Total time per run", total_times)