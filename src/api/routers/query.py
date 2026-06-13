from __future__ import annotations

import json
import re
import time
import unicodedata
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from configs.setting import config
from src.llm import get_llm
from src.tools.retrieval_tools import (
    format_docs_for_context_with_query,
    retrieve_documents,
)

router = APIRouter(prefix="/api", tags=["legal-rag"])

Strategy = Literal["auto", "dense", "hybrid", "hybrid_rerank", "graph"]
AnswerStyle = Literal["short", "normal", "detailed"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(
        default_factory=lambda: int(getattr(config.api, "default_k", 5)),
        ge=1,
        le=20,
    )
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


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


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
    Frontend đọc từng dòng một.
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
        answer = answer[len(question) :].strip(" \n:.-")

    return answer


def _style_instruction(answer_style: str) -> str:
    """
    Gắn thêm yêu cầu độ dài câu trả lời vào prompt.
    """
    if answer_style == "detailed":
        return (
            "Trả lời tương đối chi tiết, có căn cứ pháp lý, nhưng không lan man. "
            "Có thể trình bày theo các mục: Kết luận, Căn cứ, Áp dụng."
        )

    if answer_style == "normal":
        return (
            "Trả lời rõ ràng, vừa đủ ý, ưu tiên căn cứ pháp lý và kết luận trực tiếp. "
            "Không phân tích quá dài."
        )

    return (
        "Trả lời ngắn gọn, tối đa 6-8 câu. "
        "Cấu trúc nên gồm: Kết luận, Căn cứ, Áp dụng ngắn gọn. "
        "Không phân tích lan man."
    )


def _strip_accents(text: str) -> str:
    """Lowercase text and remove Vietnamese accents for robust intent checks."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).lower()


def _contains_any(text: str, phrases: list[str] | tuple[str, ...]) -> bool:
    q = _normalize_text(text)
    q_ascii = _strip_accents(q)
    for phrase in phrases:
        p = _normalize_text(phrase)
        p_ascii = _strip_accents(p)
        if p and (p in q or p_ascii in q_ascii):
            return True
    return False


def _is_procedural_question(question: str) -> bool:
    """Detect questions that need procedure-oriented answers."""
    return _contains_any(
        question,
        [
            "thủ tục", "trình tự", "hồ sơ", "giấy tờ", "nộp ở đâu",
            "cơ quan nào", "thẩm quyền", "đăng ký", "cấp lại", "trích lục",
            "xác nhận", "khai sinh", "khai tử", "hộ tịch", "thay đổi họ",
            "thay đổi tên", "cải chính", "nhận cha", "nhận mẹ", "nhận con",
            "ly hôn đơn phương", "thuận tình ly hôn", "kết hôn với người nước ngoài",
        ],
    )


def _is_cultural_phrase_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "là gì", "nghĩa là gì", "hiểu thế nào", "cha sinh mẹ dưỡng",
            "chị ngã em nâng", "anh em", "ca dao", "tục ngữ",
        ],
    )


def _is_sanction_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "xử phạt", "bị phạt", "phạt tù", "truy cứu", "vi phạm",
            "chế tài", "xử lý", "bạo lực", "đánh con", "roi vọt",
            "sống chung như vợ chồng", "ngoại tình", "đã có vợ", "đã có chồng",
        ],
    )


def _build_source_catalog(docs, limit: int = 8) -> str:
    """Build a compact source list that the LLM is allowed to cite."""
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for idx, doc in enumerate(docs[:limit], start=1):
        meta = doc.metadata or {}
        title = str(meta.get("title") or meta.get("doc_title") or "Văn bản pháp luật").strip()
        doc_id = str(meta.get("doc_id") or "").strip()
        so_ky_hieu = str(meta.get("so_ky_hieu") or "").strip()
        article = str(meta.get("article") or "").strip()
        clause = str(meta.get("clause") or "").strip()
        status = str(meta.get("tinh_trang_hieu_luc") or meta.get("status") or "").strip()
        key = (doc_id, article, clause)
        if key in seen:
            continue
        seen.add(key)

        parts = [f"[Nguồn {idx}]", title]
        if so_ky_hieu:
            parts.append(f"Số hiệu: {so_ky_hieu}")
        elif doc_id:
            parts.append(f"DocID: {doc_id}")
        if article:
            parts.append(f"Điều {article}")
        if clause:
            parts.append(f"Khoản {clause}")
        if status:
            parts.append(f"Hiệu lực: {status}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _has_legal_citation(answer: str) -> bool:
    text = _normalize_text(answer)
    return bool(
        re.search(r"\bđiều\s+\d+", text)
        or "luật" in text
        or "nghị định" in text
        or "thông tư" in text
        or "án lệ" in text
        or "bộ luật" in text
    )


def _source_brief(docs, limit: int = 3) -> str:
    items: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        meta = doc.metadata or {}
        title = str(meta.get("title") or meta.get("doc_title") or "Văn bản pháp luật").strip()
        article = str(meta.get("article") or "").strip()
        doc_id = str(meta.get("doc_id") or "").strip()
        key = (doc_id or title, article)
        if key in seen:
            continue
        seen.add(key)
        item = title
        if article:
            item += f" - Điều {article}"
        items.append(item)
        if len(items) >= limit:
            break
    return "; ".join(items)


def _guard_answer(answer: str, question: str, docs) -> str:
    """Light post-processing guardrail for final answers.

    This does not invent legal content. It only:
    - cleans repeated question markers;
    - appends retrieved-source summary if the answer has no citation;
    - softens overly broad "insufficient context" phrasing when sources exist.
    """
    answer = _clean_answer(answer, question).strip()
    if not answer:
        return answer

    has_docs = bool(docs)
    starts_with_insufficient = bool(
        re.match(
            r"^(ngữ cảnh|context|thông tin|tài liệu)\s+(hiện\s+)?(chưa|không)\s+đủ",
            _normalize_text(answer),
        )
    )

    if has_docs and starts_with_insufficient:
        answer += (
            "\n\nLưu ý: Hệ thống đã truy xuất được một số nguồn liên quan; "
            "nhận định 'chưa đủ căn cứ' chỉ áp dụng cho những chi tiết chưa xuất hiện rõ trong ngữ cảnh."
        )

    if has_docs and not _has_legal_citation(answer):
        brief = _source_brief(docs)
        if brief:
            answer += f"\n\nNguồn pháp lý đã truy xuất: {brief}."

    return answer


def _build_answer_prompt(
    question: str,
    context: str,
    answer_style: str = "short",
    source_catalog: str = "",
) -> str:
    """
    Build grounded legal-answer prompt.

    Upgrade v2:
    - tránh việc model né trả lời chỉ vì thấy thiếu một phần context;
    - ép format rõ hơn cho câu hỏi thủ tục;
    - ép viện dẫn trong phạm vi nguồn đã truy xuất;
    - phân nhánh khi câu hỏi thiếu dữ kiện.
    """
    style = _style_instruction(answer_style)
    procedural = _is_procedural_question(question)
    cultural = _is_cultural_phrase_question(question)
    sanction = _is_sanction_question(question)

    procedure_block = ""
    if procedural:
        procedure_block = """
VÌ ĐÂY LÀ CÂU HỎI CÓ TÍNH THỦ TỤC/HÀNH CHÍNH, hãy ưu tiên trình bày thêm nếu ngữ cảnh có căn cứ:
- Thẩm quyền/cơ quan giải quyết.
- Hồ sơ hoặc giấy tờ chính.
- Trình tự thực hiện ở mức khái quát.
- Lưu ý dữ kiện còn thiếu.
""".strip()

    cultural_block = ""
    if cultural:
        cultural_block = """
Nếu câu hỏi là câu nói đời thường, ca dao/tục ngữ hoặc hỏi nghĩa của cụm từ:
- Có thể giải thích ngắn gọn nghĩa thông thường trước.
- Sau đó mới nêu khía cạnh pháp lý nếu ngữ cảnh pháp lý có liên quan.
- Không ép mọi câu giải thích ngôn ngữ đời thường thành tranh chấp pháp luật.
""".strip()

    sanction_block = ""
    if sanction:
        sanction_block = """
Nếu câu hỏi hỏi về xử phạt/chế tài/phạt tù:
- Chỉ nêu mức xử phạt cụ thể khi ngữ cảnh có căn cứ rõ.
- Phân biệt trách nhiệm hành chính, dân sự, hình sự hoặc kỷ luật nếu có căn cứ.
- Nếu thiếu Bộ luật Hình sự hoặc văn bản xử phạt trong ngữ cảnh, không được khẳng định chắc chắn có/không bị phạt tù.
""".strip()

    source_block = source_catalog.strip() or "Không có danh mục nguồn riêng; chỉ viện dẫn theo NGỮ CẢNH PHÁP LÝ."

    return f"""
Bạn là trợ lý pháp lý hỗ trợ tra cứu pháp luật Việt Nam trong lĩnh vực Hôn nhân và Gia đình.

QUY TẮC BẮT BUỘC VỀ CĂN CỨ:
1. Chỉ được trả lời dựa trên NGỮ CẢNH PHÁP LÝ được cung cấp.
2. Không bịa điều luật, không bịa số điều/khoản, không suy đoán ngoài nguồn.
3. Chỉ viện dẫn những văn bản/điều luật xuất hiện trong NGỮ CẢNH hoặc DANH MỤC NGUỒN ĐƯỢC PHÉP VIỆN DẪN.
4. Nếu ngữ cảnh có đủ căn cứ cho phần chính của câu hỏi, phải trả lời phần chính trước; không được chỉ nói "ngữ cảnh chưa đủ" rồi dừng.
5. Nếu thiếu căn cứ cho một phần phụ, hãy trả lời phần có căn cứ và ghi phần còn thiếu ở mục "Lưu ý".
6. Không cam kết chắc chắn kết quả xét xử/hành chính; phải nói phụ thuộc chứng cứ, hồ sơ và cơ quan có thẩm quyền khi cần.
7. Nếu câu hỏi thiếu dữ kiện quan trọng, hãy nêu giả định hợp lý và liệt kê dữ kiện cần bổ sung.

DANH MỤC NGUỒN ĐƯỢC PHÉP VIỆN DẪN:
{source_block}

NGỮ CẢNH PHÁP LÝ:
{context}

CÂU HỎI:
{question}

YÊU CẦU TRÌNH BÀY:
- Trả lời bằng tiếng Việt, đi thẳng vào vấn đề.
- Ưu tiên cấu trúc:
  Kết luận:
  Căn cứ pháp lý:
  Trả lời/Áp dụng:
  Lưu ý:
- Với câu hỏi nhiều khả năng, hãy chia "Trường hợp 1", "Trường hợp 2".
- Không mở đầu bằng câu xin lỗi hoặc câu chung chung nếu đã có căn cứ.
- {style}
{procedure_block}
{cultural_block}
{sanction_block}

TRẢ LỜI:
""".strip()


def _doc_has_article(docs, doc_id: str, article: str) -> bool:
    for doc in docs:
        meta = doc.metadata or {}
        if str(meta.get("doc_id") or "") == doc_id and str(meta.get("article") or "") == article:
            return True
    return False


def _doc_has_any_article(docs, targets: list[tuple[str, str]]) -> bool:
    return any(_doc_has_article(docs, doc_id, article) for doc_id, article in targets)


def _maybe_fast_answer(question: str, docs) -> str | None:
    """
    Fast template cho một số câu hỏi fact chắc chắn.
    Mục tiêu:
    - Trả lời tức thì cho case rất phổ biến.
    - Giảm phụ thuộc LLM.
    - Chỉ dùng khi context đã có đúng căn cứ chính.
    """
    q = _normalize_text(question)

    # Case: con dưới 36 tháng khi ly hôn giao cho ai nuôi?
    if (
        any(
            phrase in q
            for phrase in [
                "con dưới 36 tháng",
                "con duoi 36 thang",
                "dưới 36 tháng tuổi",
                "duoi 36 thang tuoi",
            ]
        )
        and any(phrase in q for phrase in ["ly hôn", "ly hon", "nuôi", "nuoi", "giao"])
        and _doc_has_article(docs, "52_2014_QH13", "81")
    ):
        return (
            "**Kết luận:** Khi ly hôn, con dưới 36 tháng tuổi về nguyên tắc được giao cho mẹ trực tiếp nuôi.\n\n"
            "**Căn cứ:** Khoản 3 Điều 81 Luật Hôn nhân và gia đình 2014.\n\n"
            "**Áp dụng ngắn gọn:** Quyền ưu tiên này không phải tuyệt đối. Tòa án vẫn có thể quyết định khác nếu người mẹ không đủ điều kiện trực tiếp trông nom, chăm sóc, nuôi dưỡng, giáo dục con hoặc cha mẹ có thỏa thuận khác phù hợp với lợi ích của con."
        )

    # Case: điều kiện kết hôn là gì?
    if (
        any(
            phrase in q
            for phrase in [
                "điều kiện kết hôn",
                "điều kiện để kết hôn",
                "điều kiện được kết hôn",
                "kết hôn cần điều kiện gì",
                "muốn kết hôn cần",
                "tuổi kết hôn",
                "bao nhiêu tuổi được kết hôn",
            ]
        )
        and _doc_has_article(docs, "52_2014_QH13", "8")
    ):
        return (
            "**Kết luận:** Nam, nữ muốn kết hôn hợp pháp phải đáp ứng điều kiện về độ tuổi, sự tự nguyện, năng lực hành vi dân sự và không thuộc trường hợp bị cấm kết hôn.\n\n"
            "**Căn cứ:** Điều 8 và khoản 2 Điều 5 Luật Hôn nhân và gia đình 2014.\n\n"
            "**Áp dụng ngắn gọn:** Nam phải từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên; việc kết hôn do hai bên tự nguyện quyết định; không bị mất năng lực hành vi dân sự; và không thuộc các trường hợp cấm như tảo hôn, cưỡng ép/lừa dối kết hôn, người đang có vợ/chồng mà kết hôn với người khác, hoặc kết hôn giữa những người có quan hệ huyết thống/thích thuộc bị pháp luật cấm."
        )

    # Case: tiền trúng số/trúng thưởng xổ số là tài sản chung hay riêng?
    if (
        any(
            phrase in q
            for phrase in [
                "trúng số",
                "trúng thưởng xổ số",
                "tiền trúng thưởng",
                "tiền xổ số",
                "vietlott",
            ]
        )
        and any(phrase in q for phrase in ["tài sản chung", "tài sản riêng", "ly hôn", "vợ chồng"])
        and _doc_has_article(docs, "126_2014_ND_CP", "9")
    ):
        return (
            "**Kết luận:** Tiền trúng thưởng xổ số của vợ hoặc chồng trong thời kỳ hôn nhân được xem là thu nhập hợp pháp khác của vợ chồng, nên về nguyên tắc là tài sản chung của vợ chồng.\n\n"
            "**Căn cứ:** Khoản 1 Điều 9 Nghị định 126/2014/NĐ-CP và Điều 33 Luật Hôn nhân và gia đình 2014.\n\n"
            "**Áp dụng ngắn gọn:** Nếu ly hôn, khoản tiền này có thể được xem xét chia theo nguyên tắc chia tài sản chung của vợ chồng, trừ trường hợp có căn cứ pháp lý khác chứng minh đó là tài sản riêng hoặc có thỏa thuận hợp pháp khác."
        )

    # Case: điều kiện nhận con nuôi.
    if (
        _contains_any(q, ["điều kiện nhận con nuôi", "điều kiện người nhận con nuôi", "nhận con nuôi cần điều kiện gì"])
        and _doc_has_any_article(docs, [("52_2010_QH12", "14"), ("52_2010_QH12", "8")])
    ):
        return (
            "**Kết luận:** Việc nhận con nuôi phải đáp ứng điều kiện đối với cả người nhận con nuôi và người được nhận làm con nuôi.\n\n"
            "**Căn cứ:** Điều 14 và Điều 8 Luật Nuôi con nuôi 2010.\n\n"
            "**Áp dụng ngắn gọn:** Người nhận con nuôi thường phải có năng lực hành vi dân sự đầy đủ, hơn con nuôi từ 20 tuổi trở lên, có điều kiện về sức khỏe, kinh tế, chỗ ở và tư cách đạo đức tốt; đồng thời không thuộc các trường hợp bị cấm. Người được nhận làm con nuôi chủ yếu là trẻ em dưới 16 tuổi; người từ đủ 16 đến dưới 18 tuổi chỉ thuộc một số trường hợp luật định như được cha dượng, mẹ kế hoặc cô, cậu, dì, chú, bác ruột nhận làm con nuôi."
        )

    # Case: ly hôn đơn phương - điều kiện và thủ tục khái quát.
    if (
        _contains_any(q, ["ly hôn đơn phương", "đơn phương ly hôn", "ly hôn theo yêu cầu của một bên"])
        and _doc_has_article(docs, "52_2014_QH13", "56")
    ):
        return (
            "**Kết luận:** Có thể yêu cầu ly hôn đơn phương nếu có căn cứ cho thấy đời sống chung không thể kéo dài, mục đích hôn nhân không đạt được hoặc có hành vi bạo lực gia đình/vi phạm nghiêm trọng quyền, nghĩa vụ của vợ chồng.\n\n"
            "**Căn cứ:** Điều 51 và Điều 56 Luật Hôn nhân và gia đình 2014.\n\n"
            "**Áp dụng ngắn gọn:** Người yêu cầu cần chuẩn bị đơn ly hôn, giấy tờ nhân thân, giấy chứng nhận kết hôn hoặc trích lục, giấy tờ về con chung/tài sản/nợ chung nếu có, và chứng cứ chứng minh yêu cầu ly hôn. Thẩm quyền, hồ sơ chi tiết và trình tự tố tụng còn phụ thuộc nơi cư trú của bị đơn và yêu cầu cụ thể trong vụ án."
        )

    # Case: ly hôn khi không có giấy chứng nhận kết hôn.
    if (
        _contains_any(q, ["ly hôn khi không có giấy kết hôn", "không có giấy kết hôn", "mất giấy chứng nhận kết hôn", "không có đăng ký kết hôn"])
        and _contains_any(q, ["ly hôn", "chia tay", "vợ chồng"])
    ):
        return (
            "**Kết luận:** Cần phân biệt hai trường hợp: đã đăng ký kết hôn nhưng mất giấy chứng nhận, hoặc chưa từng đăng ký kết hôn.\n\n"
            "**Căn cứ pháp lý:** Luật Hôn nhân và gia đình 2014 quy định việc kết hôn phải được đăng ký theo quy định pháp luật; nam nữ không đăng ký kết hôn thì không làm phát sinh quyền, nghĩa vụ vợ chồng.\n\n"
            "**Áp dụng ngắn gọn:** Nếu đã đăng ký kết hôn nhưng mất giấy, bạn thường cần xin trích lục/ bản sao giấy chứng nhận kết hôn tại cơ quan hộ tịch để nộp hồ sơ ly hôn. Nếu chưa từng đăng ký kết hôn, Tòa án thường không giải quyết 'ly hôn' theo nghĩa chấm dứt hôn nhân hợp pháp, nhưng vẫn có thể xem xét vấn đề con chung, tài sản hoặc nghĩa vụ liên quan nếu có yêu cầu và chứng cứ."
        )

    # Case: thay đổi họ/tên cho con hoặc cải chính hộ tịch.
    if (
        _contains_any(q, ["thay đổi họ", "thay đổi tên", "đổi họ", "đổi tên", "cải chính hộ tịch"])
        and _contains_any(q, ["con", "trẻ", "khai sinh", "hộ tịch"])
        and _doc_has_any_article(docs, [("60_2014_QH13", "26"), ("123_2015_ND_CP", "7")])
    ):
        return (
            "**Kết luận:** Việc thay đổi họ, tên hoặc cải chính thông tin hộ tịch cho con là thủ tục hộ tịch và phải có căn cứ theo luật.\n\n"
            "**Căn cứ:** Điều 26 Luật Hộ tịch 2014 và Điều 7 Nghị định 123/2015/NĐ-CP nếu có trong hồ sơ/ngữ cảnh áp dụng.\n\n"
            "**Áp dụng ngắn gọn:** Với người chưa thành niên, việc thay đổi/cải chính hộ tịch thường cần sự đồng ý của cha, mẹ; nếu trẻ đã đủ độ tuổi luật định thì còn cần xem xét ý kiến của trẻ. Cơ quan hộ tịch sẽ căn cứ giấy tờ chứng minh lý do thay đổi, thông tin khai sinh hiện tại và quan hệ cha mẹ con để giải quyết."
        )

    # Case: nhận cha, mẹ, con.
    if (
        _contains_any(q, ["nhận cha", "nhận mẹ", "nhận con", "xác nhận cha con", "xác định cha cho con"])
        and _doc_has_any_article(docs, [("60_2014_QH13", "24"), ("60_2014_QH13", "25"), ("52_2014_QH13", "88")])
    ):
        return (
            "**Kết luận:** Việc nhận cha, mẹ, con có thể thực hiện theo thủ tục hộ tịch nếu có đủ căn cứ chứng minh quan hệ cha mẹ con và không có tranh chấp.\n\n"
            "**Căn cứ:** Luật Hộ tịch 2014 về đăng ký nhận cha, mẹ, con; Luật Hôn nhân và gia đình 2014 về xác định cha, mẹ, con nếu ngữ cảnh có liên quan.\n\n"
            "**Áp dụng ngắn gọn:** Người yêu cầu cần giấy tờ tùy thân, giấy tờ hộ tịch của người liên quan và chứng cứ chứng minh quan hệ cha mẹ con, ví dụ kết quả giám định ADN hoặc chứng cứ khác được cơ quan có thẩm quyền chấp nhận. Nếu có tranh chấp hoặc người liên quan đã chết, mất tích, không hợp tác, vụ việc có thể phải được Tòa án xem xét."
        )

    return None


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
            fast_answer = _maybe_fast_answer(req.question, docs)

            if fast_answer:
                answer = fast_answer
                timings["fast_answer_ms"] = (time.perf_counter() - request_t0) * 1000
            else:
                t0 = time.perf_counter()
                context = format_docs_for_context_with_query(docs, req.question)[: req.max_context_chars]
                timings["context_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                prompt = _build_answer_prompt(
                    question=req.question,
                    context=context,
                    answer_style=req.answer_style,
                    source_catalog=_build_source_catalog(docs),
                )
                timings["prompt_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                llm = get_llm()
                timings["llm_init_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                answer = _guard_answer(_llm_output_to_text(llm.invoke(prompt)), req.question, docs)
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
async def query_stream(req: QueryRequest):  # 1. THÊM TỪ KHÓA 'async'
    """
    Endpoint streaming response.
    """

    async def generate():  # 2. THÊM TỪ KHÓA 'async'
        request_t0 = time.perf_counter()
        timings: dict[str, float] = {}

        try:
            # 1. Retrieval trước
            docs, retrieval_ms = _agentic_retrieve(req.question, req.k, req.strategy)
            timings["retrieval_ms"] = retrieval_ms

            documents = _docs_payload(docs, snippet_chars=700)

            yield _json_line({
                "type": "sources",
                "documents": documents,
                "latency_ms": retrieval_ms,
            })

            # 2. Nếu tắt generate thì chỉ trả nguồn
            if not req.generate:
                answer = "Đã truy xuất nguồn pháp luật liên quan. Chế độ sinh câu trả lời đang tắt."
                timings["total_ms"] = (time.perf_counter() - request_t0) * 1000
                yield _json_line({"type": "done", "answer": answer, "timings": timings})
                return

            # 3. Fast answer cho câu hỏi fact chắc chắn
            fast_answer = _maybe_fast_answer(req.question, docs)
            if fast_answer:
                timings["total_ms"] = (time.perf_counter() - request_t0) * 1000
                yield _json_line({"type": "delta", "text": fast_answer})
                yield _json_line({"type": "done", "answer": fast_answer, "timings": timings})
                return

            # 4. Tạo context
            t0 = time.perf_counter()
            context = format_docs_for_context_with_query(docs, req.question)[: req.max_context_chars]
            timings["context_ms"] = (time.perf_counter() - t0) * 1000

            yield _json_line({"type": "status", "message": "Đang chuẩn bị ngữ cảnh pháp lý...", "timings": timings})

            # 5. Tạo prompt
            prompt = _build_answer_prompt(
                req.question,
                context,
                req.answer_style,
                source_catalog=_build_source_catalog(docs),
            )

            # 6. Khởi tạo LLM
            llm = get_llm()

            yield _json_line({"type": "status", "message": "Đang sinh câu trả lời...", "timings": timings})

            # ========================================================
            # 7. SỬA ĐỔI QUAN TRỌNG NHẤT: DÙNG ASTREAM()
            # ========================================================
            answer_parts: list[str] = []
            llm_t0 = time.perf_counter()
            first_token_sent = False

            try:
                # 3. DÙNG 'async for' VÀ 'llm.astream' THAY VÌ 'llm.stream'
                async for chunk in llm.astream(prompt): 
                    text = _chunk_to_text(chunk)

                    if not text:
                        continue

                    if not first_token_sent:
                        first_token_sent = True
                        timings["first_token_ms"] = (time.perf_counter() - request_t0) * 1000
                        yield _json_line({
                            "type": "first_token",
                            "latency_ms": timings["first_token_ms"],
                            "timings": timings,
                        })

                    answer_parts.append(text)

                    yield _json_line({
                        "type": "delta",
                        "text": text,
                    })

            except Exception as stream_exc:
                # Fallback nếu wrapper/model không hỗ trợ stream thật
                yield _json_line({"type": "status", "message": "Model không hỗ trợ stream trực tiếp..."})
                
                # Gọi đồng bộ trong trường hợp astream bị lỗi (thường rất hiếm)
                response = llm.invoke(prompt)
                text = _llm_output_to_text(response)
                
                if text:
                    answer_parts.append(text)
                    yield _json_line({"type": "delta", "text": text})

            timings["llm_generation_ms"] = (time.perf_counter() - llm_t0) * 1000

            raw_answer = "".join(answer_parts)
            answer = _guard_answer(raw_answer, req.question, docs)
            timings["total_ms"] = (time.perf_counter() - request_t0) * 1000

            yield _json_line({
                "type": "done",
                "answer": answer,
                "timings": timings,
            })

        except Exception as exc:
            yield _json_line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive", # Thêm header này để chống timeout
        },
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