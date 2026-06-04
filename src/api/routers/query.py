from __future__ import annotations

import json
import time
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from configs.setting import config
from src.llm import get_llm
from src.tools.retrieval_tools import (
    format_docs_for_context,
    generate_answer_tool,
    retrieve_documents,
)

router = APIRouter(prefix="/api", tags=["legal-rag"])

Strategy = Literal["auto", "dense", "hybrid", "hybrid_rerank", "graph"]
AnswerStyle = Literal["short", "normal", "detailed"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default_factory=lambda: int(getattr(config.api, "default_k", 5)), ge=1, le=20)
    strategy: Strategy = "auto"
    generate: bool = True
    max_context_chars: int = Field(default=4000, ge=500, le=20000)

    # Dùng cho frontend streaming, có cũng được, không có cũng không sao
    answer_style: AnswerStyle = "short"


class KeywordLookupRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    k: int = Field(default=5, ge=1, le=10)
    strategy: Strategy = "dense"


class QueryResponse(BaseModel):
    question: str
    strategy: str
    latency_ms: float
    answer: str | None
    documents: list[dict]
    timings: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class KeywordLookupResponse(BaseModel):
    keyword: str
    strategy: str
    latency_ms: float
    documents: list[dict]


def _short_text(text: str, limit: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _docs_payload(docs, snippet_chars: int = 900) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        payload.append(
            {
                "rank": idx,
                "page_content": doc.page_content,
                "snippet": _short_text(doc.page_content or "", limit=snippet_chars),
                "metadata": metadata,
                "title": metadata.get("title") or metadata.get("doc_title") or "Văn bản pháp luật",
                "doc_id": metadata.get("doc_id"),
                "so_ky_hieu": metadata.get("so_ky_hieu"),
                "article": metadata.get("article"),
                "clause": metadata.get("clause"),
                "source_url": metadata.get("source_url") or metadata.get("url"),
                "status": metadata.get("tinh_trang_hieu_luc"),
                "route": metadata.get("route"),
                "corpus_role": metadata.get("corpus_role"),
            }
        )

    return payload


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


def _chunk_to_text(chunk) -> str:
    """
    Chuẩn hóa chunk khi LLM stream.
    Hỗ trợ:
    - str
    - AIMessageChunk có .content
    - list content blocks
    - dict
    """
    if chunk is None:
        return ""

    content = getattr(chunk, "content", chunk)

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

        return "".join(parts)

    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")

    return str(content or "")


def _json_line(payload: dict[str, Any]) -> str:
    """
    Trả từng dòng JSON theo chuẩn NDJSON.
    Frontend sẽ đọc từng dòng một.
    """
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _clean_answer(answer: str, question: str) -> str:
    answer = (answer or "").strip()
    question = (question or "").strip()

    for marker in ["[TRẢ LỜI]:", "[TRẢ LỜI]", "Trả lời:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()

    if answer.lower().strip(" ?.!…") == question.lower().strip(" ?.!…"):
        return (
            "Chưa sinh được câu trả lời phù hợp. "
            "Vui lòng kiểm tra lại ngữ cảnh truy xuất hoặc prompt generation."
        )

    if answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip(" \n:.-")

    return answer


def _style_instruction(answer_style: str) -> str:
    """
    Gắn thêm yêu cầu độ dài câu trả lời vào prompt.
    Không bắt buộc, nhưng giúp streaming nhanh và gọn hơn.
    """
    if answer_style == "detailed":
        return (
            "\n\nYêu cầu trình bày: Trả lời tương đối chi tiết, có căn cứ pháp lý, "
            "nhưng không lan man."
        )

    if answer_style == "normal":
        return (
            "\n\nYêu cầu trình bày: Trả lời rõ ràng, vừa đủ ý, ưu tiên căn cứ pháp lý "
            "và kết luận trực tiếp."
        )

    return (
        "\n\nYêu cầu trình bày: Trả lời ngắn gọn, tối đa 6-8 câu. "
        "Cấu trúc nên gồm: Kết luận, Căn cứ, Áp dụng ngắn gọn. "
        "Không phân tích lan man."
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    request_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    docs, retrieval_ms = _agentic_retrieve(req.question, req.k, req.strategy)
    timings["retrieval_ms"] = retrieval_ms

    answer = None
    error = None

    if req.generate:
        try:
            t0 = time.perf_counter()
            context = format_docs_for_context(docs)[: req.max_context_chars]
            timings["context_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            prompt = generate_answer_tool.invoke(
                {
                    "query": req.question,
                    "context": context,
                }
            )
            prompt = str(prompt) + _style_instruction(req.answer_style)
            timings["prompt_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            llm = get_llm()
            timings["llm_init_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            answer = _clean_answer(_llm_output_to_text(llm.invoke(prompt)), req.question)
            timings["llm_generation_ms"] = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            answer = (
                "Không thể gọi mô hình sinh câu trả lời ở thời điểm này. "
                "Hệ thống vẫn trả về các nguồn pháp luật đã truy xuất để bạn kiểm tra."
            )

    total_ms = (time.perf_counter() - request_t0) * 1000
    timings["total_ms"] = total_ms

    return QueryResponse(
        question=req.question,
        strategy=req.strategy,
        latency_ms=retrieval_ms,
        answer=answer,
        documents=_docs_payload(docs),
        timings=timings,
        error=error,
    )


@router.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Endpoint streaming response.

    URL đầy đủ là:
        /api/query/stream

    Vì router đã có prefix="/api", nên decorator chỉ cần:
        @router.post("/query/stream")
    """

    def generate():
        request_t0 = time.perf_counter()
        timings: dict[str, float] = {}

        try:
            # 1. Retrieval trước
            docs, retrieval_ms = _agentic_retrieve(req.question, req.k, req.strategy)
            timings["retrieval_ms"] = retrieval_ms

            documents = _docs_payload(docs, snippet_chars=700)

            yield _json_line(
                {
                    "type": "sources",
                    "documents": documents,
                    "latency_ms": retrieval_ms,
                }
            )

            # 2. Nếu người dùng tắt generate thì chỉ trả nguồn
            if not req.generate:
                answer = (
                    "Đã truy xuất nguồn pháp luật liên quan. "
                    "Chế độ sinh câu trả lời đang tắt."
                )

                total_ms = (time.perf_counter() - request_t0) * 1000
                timings["total_ms"] = total_ms

                yield _json_line(
                    {
                        "type": "done",
                        "answer": answer,
                        "timings": timings,
                    }
                )
                return

            # 3. Tạo context
            t0 = time.perf_counter()
            context = format_docs_for_context(docs)[: req.max_context_chars]
            timings["context_ms"] = (time.perf_counter() - t0) * 1000

            # 4. Tạo prompt
            t0 = time.perf_counter()
            prompt = generate_answer_tool.invoke(
                {
                    "query": req.question,
                    "context": context,
                }
            )
            prompt = str(prompt) + _style_instruction(req.answer_style)
            timings["prompt_ms"] = (time.perf_counter() - t0) * 1000

            # 5. Khởi tạo LLM
            t0 = time.perf_counter()
            llm = get_llm()
            timings["llm_init_ms"] = (time.perf_counter() - t0) * 1000

            # 6. Stream LLM
            answer_parts: list[str] = []
            llm_t0 = time.perf_counter()

            try:
                for chunk in llm.stream(prompt):
                    text = _chunk_to_text(chunk)

                    if not text:
                        continue

                    answer_parts.append(text)

                    yield _json_line(
                        {
                            "type": "delta",
                            "text": text,
                        }
                    )

            except Exception:
                # Fallback nếu wrapper/model không hỗ trợ stream
                response = llm.invoke(prompt)
                text = _llm_output_to_text(response)

                if text:
                    answer_parts.append(text)

                    yield _json_line(
                        {
                            "type": "delta",
                            "text": text,
                        }
                    )

            timings["llm_generation_ms"] = (time.perf_counter() - llm_t0) * 1000

            raw_answer = "".join(answer_parts)
            answer = _clean_answer(raw_answer, req.question)

            total_ms = (time.perf_counter() - request_t0) * 1000
            timings["total_ms"] = total_ms

            yield _json_line(
                {
                    "type": "done",
                    "answer": answer,
                    "timings": timings,
                }
            )

        except Exception as exc:
            yield _json_line(
                {
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@router.post("/keyword/lookup", response_model=KeywordLookupResponse)
def keyword_lookup(req: KeywordLookupRequest):
    keyword = " ".join(req.keyword.split()).strip()

    t0 = time.perf_counter()
    docs = retrieve_documents(keyword, k=req.k, strategy=req.strategy)
    latency_ms = (time.perf_counter() - t0) * 1000

    return KeywordLookupResponse(
        keyword=keyword,
        strategy=req.strategy,
        latency_ms=latency_ms,
        documents=_docs_payload(docs, snippet_chars=700),
    )