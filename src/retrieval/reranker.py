import re
from langchain_core.documents import Document
from underthesea import word_tokenize

# Loại bỏ các stop-words gây nhiễu ngữ nghĩa thông thường, GIỮ LẠI các từ định vị như "điều", "khoản"
LEGAL_STOPWORDS = {
    "là", "gì", "của", "được", "theo", "về", "và", "trong", "các", 
    "những", "cho", "đến", "nào", "đã", "đang", "sẽ", "có", "thì", "mà"
}

def _extract_law_identifiers(text: str) -> set[str]:
    """
    Sử dụng Regex để bóc tách chính xác các thực thể số hiệu pháp lý quan trọng (Ví dụ: điều_8, khoản_2).
    """
    text_lower = text.lower()
    identifiers = set()
    
    # Tìm cấu trúc "điều X" (X là số)
    dieu_matches = re.findall(r"\bđiều\s+(\d+)", text_lower)
    for m in dieu_matches:
        identifiers.add(f"dieu_{m}")
        
    # Tìm cấu trúc "khoản Y" (Y là số)
    khoan_matches = re.findall(r"\bkhoản\s+(\d+)", text_lower)
    for m in khoan_matches:
        identifiers.add(f"khoan_{m}")
        
    return identifiers

def _tokenize_vietnamese(text: str) -> set[str]:
    """
    Tách từ ghép tiếng Việt bằng underthesea kết hợp giữ lại cấu trúc từ đơn sạch.
    """
    if not text:
        return set()
    
    text = text.lower()
    # Tách từ ghép nối bằng dấu gạch dưới '_'
    segmented_text = word_tokenize(text, format="text")
    tokens = re.findall(r"\w+", segmented_text, flags=re.UNICODE)
    
    # Lọc bỏ stopwords thông thường
    return {t for t in tokens if t not in LEGAL_STOPWORDS}

def _score(query: str, doc: Document) -> float:
    """
    Hàm chấm điểm nâng cao kết hợp Overlap Score và Trọng số Thực thể Pháp luật cứng.
    """
    query_clean = query.lower()
    page_content_clean = doc.page_content.lower()
    
    # 1. Trích xuất và so khớp thực thể pháp lý cứng (Điều / Khoản)
    query_ids = _extract_law_identifiers(query_clean)
    doc_ids = _extract_law_identifiers(page_content_clean)
    
    identity_bonus = 0.0
    if query_ids:
        matched_ids = query_ids & doc_ids
        # Nếu trùng khớp chính xác số Điều luật mà người dùng hỏi, cộng điểm thưởng cực lớn
        if matched_ids:
            identity_bonus += 0.6 * len(matched_ids)
            
    # 2. Tính toán điểm ngữ nghĩa bằng Overlap Score (Chống trùng lặp bias chunk ngắn của Jaccard)
    query_tokens = _tokenize_vietnamese(query_clean)
    doc_tokens = _tokenize_vietnamese(page_content_clean)
    
    if not query_tokens:
        return identity_bonus

    intersection = query_tokens & doc_tokens
    
    # Công thức Overlap Score cải tiến: Số từ khớp / Căn bậc hai độ dài câu hỏi nhằm chuẩn hóa điểm số
    overlap_score = len(intersection) / (len(query_tokens) ** 0.5)
    
    # 3. Tính điểm thưởng dựa trên thông tin làm giàu Metadata nguồn
    meta = doc.metadata or {}
    title = str(meta.get("title", "")).lower()
    so_ky_hieu = str(meta.get("so_ky_hieu", "")).lower()
    
    metadata_bonus = 0.0
    if so_ky_hieu and so_ky_hieu in query_clean:
        metadata_bonus += 0.5  # Khớp chính xác số hiệu văn bản (Ví dụ: 52/2014/QH13)
        
    if title:
        title_tokens = _tokenize_vietnamese(title)
        if query_tokens & title_tokens:
            metadata_bonus += 0.15

    # Tổng điểm cuối cùng bằng tổng hòa các trọng số chuyên biệt
    return overlap_score + identity_bonus + metadata_bonus

def rerank(
    query: str,
    docs: list[Document],
    k: int = 5,
) -> list[Document]:
    """
    Rerank danh sách văn bản dựa trên thuật toán Overlap Score phối hợp Regex Thực thể Pháp lý.
    """
    if not docs:
        return []

    scored_docs = [
        (_score(query, doc), doc)
        for doc in docs
    ]

    # Sắp xếp danh sách giảm dần theo điểm số tổng hợp
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:k]]