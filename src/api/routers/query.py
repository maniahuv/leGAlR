from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from configs.setting import config
from src.llm import get_llm
from src.tools.retrieval_tools import format_docs_for_context, generate_answer_tool, retrieve_documents

router = APIRouter(prefix="/api", tags=["legal-rag"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default_factory=lambda: int(getattr(config.api, "default_k", 5)), ge=1, le=20)
    strategy: Literal["auto", "dense", "hybrid", "hybrid_rerank", "graph"] = "auto"
    generate: bool = True


class QueryResponse(BaseModel):
    question: str
    strategy: str
    latency_ms: float
    answer: str | None
    documents: list[dict]


def _docs_payload(docs):
    return [{"page_content": d.page_content, "metadata": d.metadata or {}} for d in docs]


def _agentic_retrieve(question: str, k: int, strategy: str = "auto"):
    t0 = time.perf_counter()
    docs = retrieve_documents(question, k=k, strategy=strategy)
    latency_ms = (time.perf_counter() - t0) * 1000
    return docs, latency_ms




def _llm_output_to_text(output) -> str:
    """Normalize LangChain/Gemini outputs to plain text."""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p.strip())
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content or "")

def _clean_answer(answer: str, question: str) -> str:
    answer = (answer or "").strip()
    question = (question or "").strip()

    # Xóa nhãn thừa nếu model sinh ra
    for marker in ["[TRẢ LỜI]:", "[TRẢ LỜI]", "Trả lời:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    # Nếu model lặp lại nguyên câu hỏi thì coi là lỗi generation
    if answer.lower().strip(" ?.!") == question.lower().strip(" ?.!"):
        return (
            "Chưa sinh được câu trả lời phù hợp. "
            "Vui lòng kiểm tra lại ngữ cảnh truy xuất hoặc prompt generation."
        )

    # Nếu câu trả lời mở đầu bằng câu hỏi, cắt phần câu hỏi đi
    if answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip(" \n:.-")

    return answer


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    docs, latency_ms = _agentic_retrieve(req.question, req.k, req.strategy)
    answer = None
    if req.generate:
        context = format_docs_for_context(docs)
        prompt = generate_answer_tool.invoke({"query": req.question, "context": context})
        llm = get_llm()
        answer = _clean_answer(_llm_output_to_text(llm.invoke(prompt)), req.question)
    return QueryResponse(
        question=req.question,
        strategy=req.strategy,
        latency_ms=latency_ms,
        answer=answer,
        documents=_docs_payload(docs),
    )
