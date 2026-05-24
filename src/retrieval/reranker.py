import re
from langchain_core.documents import Document


def _tokenize(text: str) -> set[str]:
    """
    Tokenize đơn giản cho tiếng Việt.
    """
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return set(tokens)


def _score(query: str, doc: Document) -> float:
    """
    Chấm điểm document dựa trên độ trùng token với query.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc.page_content)

    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = query_tokens & doc_tokens

    return len(overlap) / len(query_tokens)


def rerank(
    query: str,
    docs: list[Document],
    k: int = 5,
) -> list[Document]:
    """
    Rerank documents theo độ liên quan đơn giản.
    Không cần model, phù hợp bản đồ án chạy CPU.
    """
    scored_docs = [
        (_score(query, doc), doc)
        for doc in docs
    ]

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:k]]