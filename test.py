from __future__ import annotations

import sys
from pathlib import Path

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


if __name__ == "__main__":
    question = (
        "Trường hợp hai vợ chồng ly hôn giành quyền nuôi con, "
        "nếu người chồng có điều kiện kinh tế tốt hơn nhưng người vợ có thời gian "
        "chăm sóc con nhiều hơn thì tòa án sẽ ưu tiên quyền nuôi con cho ai?"
    )

    docs = retrieve_documents(question, k=5, strategy="hybrid_rerank")
    context = format_docs_for_context(docs)

    print("\n================= RETRIEVED CONTEXT =================")
    print(context[:4000])

    prompt = generate_answer_tool.invoke({
        "query": question,
        "context": context,
    })

    llm = get_llm()
    response = llm.invoke(prompt)
    answer = clean_answer(response, question)

    print("\n================= ANSWER =================")
    print(answer)