import re
from langchain_core.documents import Document

# Danh sách từ dừng pháp lý cơ bản có tần suất xuất hiện quá cao gây nhiễu điểm số
LEGAL_STOPWORDS = {
    "là", "gì", "của", "được", "theo", "quy", "định", "tại", "về", 
    "và", "trong", "các", "những", "số", "điều", "khoản", "điểm", 
    "cho", "đến", "nào", "đã", "đang", "sẽ", "có", "thì", "mà"
}

def _tokenize(text: str) -> set[str]:
    """
    Tokenize đơn giản cho tiếng Việt và thực hiện lọc nhiễu từ dừng.
    """
    if not text:
        return set()
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    # Loại bỏ bớt các stop-words để tránh việc các đoạn văn bản chứa từ chung chung leo top
    return {t for t in tokens if t not in LEGAL_STOPWORDS}

def _score(query: str, doc: Document) -> float:
    """
    Chấm điểm document dựa trên độ trùng lặp Jaccard Similarity đã lọc nhiễu.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc.page_content)

    if not query_tokens:
        return 0.0

    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens

    if not union:
        return 0.0

    # Tính toán chỉ số tương đồng Jaccard (chuẩn hóa theo độ dài tập hợp từ)
    return len(intersection) / len(union)

def rerank(
    query: str,
    docs: list[Document],
    k: int = 5,
) -> list[Document]:
    """
    Rerank documents theo độ tương đồng Jaccard cải tiến.
    Phù hợp cho môi trường chạy CPU, giúp tối ưu hóa thứ tự hiển thị chunk chuẩn xác.
    """
    if not docs:
        return []

    scored_docs = [
        (_score(query, doc), doc)
        for doc in docs
    ]

    # Sắp xếp giảm dần theo điểm số tương đồng
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:k]]