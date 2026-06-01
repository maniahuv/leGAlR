from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

LEGAL_STOPWORDS = {
    "là", "gì", "của", "được", "theo", "về", "và", "trong", "các", "những", "cho",
    "đến", "nào", "đã", "đang", "sẽ", "có", "thì", "mà", "một", "như", "khi", "nếu",
    "ở", "bị", "bởi", "với", "cần", "phải", "hay", "hoặc", "do", "để", "từ", "này",
}

RELATION_QUERY_PATTERNS = [
    # Chỉ route sang Graph khi hỏi quan hệ giữa văn bản.
    # Không coi riêng "còn hiệu lực"/"hết hiệu lực" là graph intent.
    r"\bsửa\s+đổi\b",
    r"\bbổ\s+sung\b",
    r"\bthay\s+thế\b",
    r"\bbãi\s+bỏ\b",
    r"\bhủy\s+bỏ\b",
    r"\bbị\s+sửa\s+đổi\b",
    r"\bbị\s+bổ\s+sung\b",
    r"\bbị\s+thay\s+thế\b",
    r"\bbị\s+bãi\s+bỏ\b",
    r"\bvăn\s+bản\s+(sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ)\b",
    r"\bvăn\s+bản\s+nào\s+(sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ)\b",
    r"\bvăn\s+bản\s+hướng\s+dẫn\b",
    r"\bnghị\s+định\s+hướng\s+dẫn\b",
    r"\bthông\s+tư\s+hướng\s+dẫn\b",
    r"\bhướng\s+dẫn\s+thi\s+hành\b",
    r"\bquy\s+định\s+chi\s+tiết\b",
    r"\bdẫn\s+chiếu\b",
    r"\btham\s+chiếu\b",
    r"\bquan\s+hệ\s+giữa\b",
    r"\bvăn\s+bản\s+nào\s+liên\s+quan\b",
    r"\bvăn\s+bản\s+liên\s+quan\s+đến\b",
]

# Backward-compatible name for older imports/tests.
RELATION_QUERY_KEYWORDS = RELATION_QUERY_PATTERNS

EXACT_QUERY_PATTERNS = [
    r"\bđiều\s+\d+[a-zA-Z]?\b",
    r"\bkhoản\s+\d+\b",
    r"\bđiểm\s+[a-z]\b",
    r"\b\d+\s*/\s*\d{4}\s*/\s*[\w\-Đđ]+\b",
    r"\b(luật|nghị định|thông tư|nghị quyết|quyết định)\b",
    r"\b(qh\d+|nđ-cp|nd-cp|ttlt|tt-btp|ubtvqh)\b",
]

CURRENT_LAW_KEYWORDS = [
    "hiện hành", "hiện nay", "bây giờ", "đang áp dụng", "còn hiệu lực", "theo pháp luật việt nam",
    "quy định hiện tại", "mới nhất", "áp dụng hiện nay",
]

FAMILY_LAW_TERMS = [
    "hôn nhân", "gia đình", "ly hôn", "kết hôn", "vợ chồng", "chồng", "vợ",
    "nuôi con", "quyền nuôi con", "trực tiếp nuôi con", "con dưới 36 tháng", "con từ đủ 7 tuổi",
    "nguyện vọng của con", "cấp dưỡng", "mức cấp dưỡng", "con chung", "con riêng",
    "tài sản chung", "tài sản riêng", "chế độ tài sản", "chia tài sản", "nghĩa vụ chung",
    "tảo hôn", "kết hôn giả tạo", "cưỡng ép kết hôn", "cấm kết hôn",
    "mang thai hộ", "sinh con", "cha mẹ", "xác định cha mẹ con", "nhận cha", "nhận mẹ",
    "nhận con", "giám hộ", "nuôi con nuôi", "hộ tịch", "đăng ký kết hôn", "khai sinh",
    "bạo lực gia đình", "trẻ em",
]

TOPIC_TERMS: dict[str, list[str]] = {
    "custody": [
        "quyền nuôi con", "nuôi con", "trực tiếp nuôi con", "con dưới 36 tháng", "con từ đủ 7 tuổi",
        "nguyện vọng của con", "giao con", "chăm sóc con", "trông nom", "chăm sóc", "giáo dục con",
    ],
    "support": ["cấp dưỡng", "mức cấp dưỡng", "nghĩa vụ cấp dưỡng", "tiền cấp dưỡng"],
    "property": ["tài sản chung", "tài sản riêng", "chia tài sản", "chế độ tài sản", "nghĩa vụ chung", "nợ chung"],
    "marriage": ["kết hôn", "điều kiện kết hôn", "tảo hôn", "cấm kết hôn", "kết hôn giả tạo", "cưỡng ép kết hôn"],
    "divorce": ["ly hôn", "thuận tình ly hôn", "đơn phương ly hôn", "yêu cầu ly hôn"],
    "civil_status": ["hộ tịch", "đăng ký kết hôn", "khai sinh", "đăng ký khai sinh"],
    "domestic_violence": ["bạo lực gia đình", "phòng chống bạo lực gia đình"],
}

NOISE_TERMS_BY_TOPIC: dict[str, list[str]] = {
    "custody": ["đất đai", "quyền sử dụng đất", "nhà ở", "bất động sản", "thừa kế", "doanh nghiệp"],
    "support": ["đất đai", "quyền sử dụng đất", "nhà ở", "bất động sản", "thuế"],
    "marriage": ["đất đai", "bất động sản", "doanh nghiệp", "đấu thầu"],
}

LEGAL_DOC_TYPES = ["luật", "nghị định", "thông tư", "nghị quyết", "quyết định"]

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> set[str]:
    text = normalize_text(text)
    if word_tokenize is not None:
        text = word_tokenize(text, format="text")
    tokens = re.findall(r"[\w/\.\-]+", text, flags=re.UNICODE)
    return {t for t in tokens if t and t not in LEGAL_STOPWORDS}


def is_relation_query(query: str) -> bool:
    q = normalize_text(query)
    return any(re.search(pattern, q, flags=re.IGNORECASE | re.UNICODE) for pattern in RELATION_QUERY_PATTERNS)


def is_exact_query(query: str) -> bool:
    q = normalize_text(query)
    return any(re.search(p, q, flags=re.IGNORECASE | re.UNICODE) for p in EXACT_QUERY_PATTERNS)


def is_current_law_query(query: str) -> bool:
    q = normalize_text(query)
    return any(kw in q for kw in CURRENT_LAW_KEYWORDS)


def is_family_domain_query(query: str) -> bool:
    q = normalize_text(query)
    return any(term in q for term in FAMILY_LAW_TERMS)


def infer_topics(text: str) -> set[str]:
    q = normalize_text(text)
    return {topic for topic, terms in TOPIC_TERMS.items() if any(term in q for term in terms)}


def should_rerank(query: str) -> bool:
    return is_exact_query(query) or is_relation_query(query) or is_family_domain_query(query)


def route_query(query: str) -> str:
    """Return the retrieval route for strategy='auto'."""
    return "graph" if is_relation_query(query) else "hybrid_rerank"


def extract_identifiers(text: str) -> set[str]:
    t = normalize_text(text)
    ids: set[str] = set()
    for m in re.findall(r"\bđiều\s+(\d+[a-zA-Z]?)\b", t, flags=re.UNICODE):
        ids.add(f"dieu_{m.lower()}")
    for m in re.findall(r"\bkhoản\s+(\d+)\b", t, flags=re.UNICODE):
        ids.add(f"khoan_{m}")
    for m in re.findall(r"\bđiểm\s+([a-z])\b", t, flags=re.UNICODE):
        ids.add(f"diem_{m.lower()}")
    for m in re.findall(r"\b\d+\s*/\s*\d{4}\s*/\s*[\w\-Đđ]+\b", t, flags=re.UNICODE):
        ids.add(re.sub(r"\s+", "", m).lower())
    return ids


def legal_token_expansion(text: str) -> list[str]:
    """Extra exact-match tokens for BM25 query/doc texts."""
    ids = extract_identifiers(text)
    out = list(ids)
    q = normalize_text(text)
    if "luật hôn nhân" in q and "gia đình" in q:
        out.extend(["luat_hon_nhan_gia_dinh", "hngd"])
    if "52/2014" in q or "52 / 2014" in q:
        out.extend(["52_2014_qh13", "luat_hngd_2014"])
    if "22/2000" in q or "22 / 2000" in q:
        out.extend(["22_2000_qh10", "luat_hngd_2000"])
    for topic in infer_topics(text):
        out.append(f"topic_{topic}")
    return out


def metadata_text(doc: Document | dict) -> str:
    meta = doc if isinstance(doc, dict) else (doc.metadata or {})
    fields = [
        "title", "so_ky_hieu", "loai_van_ban", "linh_vuc", "nganh", "tinh_trang_hieu_luc",
        "article", "clause", "co_quan_ban_hanh", "ngay_ban_hanh", "ngay_co_hieu_luc", "ngay_het_hieu_luc",
    ]
    text = " ".join(str(meta.get(k, "") or "") for k in fields)
    article = str(meta.get("article", "") or "").strip()
    clause = str(meta.get("clause", "") or "").strip()
    so = normalize_text(str(meta.get("so_ky_hieu", "") or ""))
    extras: list[str] = []
    if article:
        extras.extend([f"Điều {article}", f"dieu_{article}"])
    if clause:
        extras.extend([f"Khoản {clause}", f"khoan_{clause}"])
    if "52/2014" in so:
        extras.extend(["luat_hngd_2014", "52_2014_qh13"])
    if "22/2000" in so:
        extras.extend(["luat_hngd_2000", "22_2000_qh10"])
    return " ".join([text, *extras])


def status_score(meta: dict, query: str = "") -> float:
    status = normalize_text(str(meta.get("tinh_trang_hieu_luc", "") or ""))
    current_sensitive = is_current_law_query(query) or is_family_domain_query(query)
    if "còn hiệu lực" in status:
        return 0.30 if current_sensitive else 0.18
    if "hết hiệu lực" in status:
        return -0.70 if current_sensitive else -0.45
    if "ngưng hiệu lực" in status or "đình chỉ" in status:
        return -0.45 if current_sensitive else -0.25
    return 0.0


def _title_overlap_score(query: str, title: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(title)
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    # Legal titles are long; use a conservative cap to avoid title-only domination.
    return min(0.35, 0.055 * overlap)


def _phrase_score(query: str, doc_text: str) -> float:
    q = normalize_text(query)
    d = normalize_text(doc_text)
    score = 0.0
    for term in FAMILY_LAW_TERMS:
        if term in q and term in d:
            score += 0.16 if " " in term else 0.06
    return min(score, 0.95)


def _topic_score(query: str, doc_text: str) -> float:
    q_topics = infer_topics(query)
    if not q_topics:
        return 0.0
    d = normalize_text(doc_text)
    score = 0.0
    for topic in q_topics:
        if any(term in d for term in TOPIC_TERMS.get(topic, [])):
            score += 0.35
        if any(term in d for term in NOISE_TERMS_BY_TOPIC.get(topic, [])):
            score -= 0.18
    return max(-0.45, min(score, 1.05))


def _legal_doc_score(query: str, meta: dict) -> float:
    q = normalize_text(query)
    title = normalize_text(str(meta.get("title", "") or ""))
    so = normalize_text(str(meta.get("so_ky_hieu", "") or ""))
    loai = normalize_text(str(meta.get("loai_van_ban", "") or ""))
    score = 0.0

    if so and re.sub(r"\s+", "", so) in re.sub(r"\s+", "", q):
        score += 1.00
    if "hôn nhân" in q and "gia đình" in q and "hôn nhân" in title and "gia đình" in title:
        score += 0.70
    if "2014" in q and ("52/2014" in so or "52 / 2014" in so):
        score += 0.90
    if "2000" in q and ("22/2000" in so or "22 / 2000" in so):
        score += 0.55
    if "luật" in q and "luật" in loai:
        score += 0.20
    return score



def metadata_direct_score(query: str, doc: Document) -> float:
    """High-precision score using only metadata fields.

    This is intentionally separated from metadata_signal_score because hybrid fusion
    needs a safe metadata/title recall channel that does not give every currently
    effective document a positive score.
    """
    meta = doc.metadata or {}
    q = normalize_text(query)
    q_compact = re.sub(r"\s+", "", q)
    title = normalize_text(str(meta.get("title", "") or ""))
    title_compact = re.sub(r"\s+", "", title)
    so = normalize_text(str(meta.get("so_ky_hieu", "") or ""))
    so_compact = re.sub(r"\s+", "", so)
    loai = normalize_text(str(meta.get("loai_van_ban", "") or ""))
    linh_vuc = normalize_text(str(meta.get("linh_vuc", "") or ""))
    nganh = normalize_text(str(meta.get("nganh", "") or ""))
    article = str(meta.get("article", "") or "").strip()
    clause = str(meta.get("clause", "") or "").strip()

    score = 0.0

    # Exact legal document number is the strongest metadata signal.
    if so_compact and so_compact in q_compact:
        score += 4.0

    # Exact/near-exact title query. This restores metadata-title-query recall.
    q_tokens = tokenize(query)
    title_tokens = tokenize(title)
    if q_tokens and title_tokens:
        overlap = len(q_tokens & title_tokens)
        coverage_q = overlap / max(len(q_tokens), 1)
        coverage_title = overlap / max(len(title_tokens), 1)
        if coverage_q >= 0.55 or coverage_title >= 0.45:
            score += 1.8 * coverage_q + 0.8 * coverage_title
        elif overlap >= 3:
            score += 0.35 * overlap

    # Title substring is rare but very reliable for generated metadata test cases.
    if title_compact and len(title_compact) >= 16:
        if title_compact in q_compact or q_compact in title_compact:
            score += 2.0

    # Article/clause exact metadata. Useful for ArticleHit and legal exact facts.
    if article and re.search(rf"\bđiều\s+{re.escape(article)}\b", q, flags=re.UNICODE):
        score += 2.2
    if clause and re.search(rf"\bkhoản\s+{re.escape(clause)}\b", q, flags=re.UNICODE):
        score += 0.9

    # Domain/category match should help, but not dominate.
    if "hôn nhân" in q and "gia đình" in q:
        if "hôn nhân" in title and "gia đình" in title:
            score += 1.1
        if "hôn nhân" in linh_vuc or "gia đình" in linh_vuc or "hôn nhân" in nganh or "gia đình" in nganh:
            score += 0.35

    # Document type and year are weak tie-breakers.
    for doc_type in LEGAL_DOC_TYPES:
        if doc_type in q and doc_type in loai:
            score += 0.25
            break
    years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", q))
    if years:
        meta_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", " ".join([
            str(meta.get("ngay_ban_hanh", "") or ""),
            str(meta.get("ngay_co_hieu_luc", "") or ""),
            str(meta.get("so_ky_hieu", "") or ""),
            title,
        ])))
        if years & meta_years:
            score += 0.25

    # Status is only a tie-breaker here, never a recall criterion by itself.
    status = normalize_text(str(meta.get("tinh_trang_hieu_luc", "") or ""))
    if score > 0:
        if "còn hiệu lực" in status:
            score += 0.08
        elif "hết hiệu lực" in status:
            score -= 0.12

    return score

def metadata_signal_score(query: str, doc: Document) -> float:
    meta = doc.metadata or {}
    q = normalize_text(query)
    doc_text = f"{doc.page_content or ''} {metadata_text(meta)}"

    score = 0.0
    score += _legal_doc_score(query, meta)
    score += _title_overlap_score(query, str(meta.get("title", "") or ""))

    # Direct metadata matching is deliberately strong for legal identifiers:
    # title, document number, year, Article/Clause. The previous conservative
    # title score protected semantic queries but caused metadata-title-query to
    # fall sharply. This signal is safe because metadata_direct_score only
    # becomes positive when the query and metadata overlap directly; validity
    # status is only a tie-breaker inside that function.
    direct_meta = metadata_direct_score(query, doc)
    if direct_meta > 0:
        score += 0.85 * direct_meta

    score += status_score(meta, query)
    score += _phrase_score(query, doc_text)
    score += _topic_score(query, doc_text)

    article = str(meta.get("article", "") or "").strip()
    clause = str(meta.get("clause", "") or "").strip()
    if article and re.search(rf"\bđiều\s+{re.escape(article)}\b", q, flags=re.UNICODE):
        score += 1.35
    if clause and re.search(rf"\bkhoản\s+{re.escape(clause)}\b", q, flags=re.UNICODE):
        score += 0.55

    q_ids = extract_identifiers(query)
    d_ids = extract_identifiers(doc_text)
    if q_ids:
        score += 0.55 * len(q_ids & d_ids)
        missing = q_ids - d_ids
        # Penalize only exact article/legal-code misses, not all metadata misses.
        if any(x.startswith("dieu_") for x in missing):
            score -= 0.30

    if meta.get("graph_distance"):
        try:
            distance = max(int(meta.get("graph_distance") or 1), 1)
            score += 0.18 / distance
        except Exception:
            score += 0.08

    return score
