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

LOTTERY_PROPERTY_KEYWORDS = [
    "trúng số",
    "trúng xổ số",
    "trúng thưởng xổ số",
    "tiền trúng số",
    "tiền trúng thưởng",
    "tiền thưởng",
    "thu nhập hợp pháp khác",
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

# ---------------------------------------------------------------------------
# Domain-aware legal intent signals
# ---------------------------------------------------------------------------
# These rules are intent-level rules, not one-off hard-coded questions. They
# map common family-law question intents to the legal document/article that is
# usually the primary authority. The goal is to make Hybrid/Hybrid_Rerank
# preserve Dense's recall while improving ArticleHit and reducing noise from
# large supporting codes/decrees.

CORE_DOC_ID = "52_2014_QH13"

DOC_TOPIC_HINTS: dict[str, list[str]] = {
    "civil_status": ["60_2014_QH13", "123_2015_ND_CP", "04_2020_TT_BTP"],
    "adoption": ["52_2010_QH12", "19_2011_ND_CP", "24_2019_ND_CP"],
    "domestic_violence": ["13_2022_QH15", "76_2023_ND_CP"],
    "sanction": ["82_2020_ND_CP", "117_2024_ND_CP"],
    "procedure": ["92_2015_QH13"],
    "civil_law": ["91_2015_QH13"],
}

INTENT_ARTICLE_BOOSTS: dict[str, dict[str, dict[str, float]]] = {
    "marriage_conditions": {CORE_DOC_ID: {"8": 8.0, "5": 1.5, "9": 1.0}},
    "prohibited_marriage_family": {CORE_DOC_ID: {"5": 8.0, "8": 1.0}},
    "illegal_marriage_cancel": {CORE_DOC_ID: {"10": 7.0, "11": 3.0, "12": 2.0, "8": 1.5}},
    "voluntary_divorce": {CORE_DOC_ID: {"55": 9.0, "54": 2.0}, "92_2015_QH13": {"397": 2.5, "29": 1.0}},
    "unilateral_divorce": {CORE_DOC_ID: {"56": 9.0, "51": 1.5}},
    "divorce_right": {CORE_DOC_ID: {"51": 8.0, "55": 1.0, "56": 1.0}},
    "custody_initial_decision": {CORE_DOC_ID: {"81": 10.0, "82": 1.5, "83": 1.0, "84": 1.0}},
    "custody_obligation": {CORE_DOC_ID: {"82": 10.0, "81": 2.0}},
    "custody_visitation": {CORE_DOC_ID: {"83": 10.0, "82": 2.0}},
    "custody_change": {CORE_DOC_ID: {"84": 10.0, "81": 2.0}, "92_2015_QH13": {"39": 2.0}},
    "child_support": {CORE_DOC_ID: {"110": 8.0, "116": 8.0, "117": 5.0, "118": 5.0}},
    "common_property": {CORE_DOC_ID: {"33": 9.0, "35": 4.0, "34": 2.0}},
    "private_property": {CORE_DOC_ID: {"43": 9.0, "44": 3.0}},
    "property_agreement": {CORE_DOC_ID: {"38": 8.0, "39": 3.0, "40": 2.0, "41": 2.0, "42": 5.0}},
    "property_division_divorce": {CORE_DOC_ID: {"59": 9.0, "60": 3.0, "61": 3.0, "62": 3.0, "63": 3.0, "64": 3.0}, "01_2016_TTLT_TANDTC_VKSNDTC_BTP": {"7": 5.0}},
    "civil_status_marriage_registration": {"60_2014_QH13": {"18": 7.0, "7": 2.0}, "123_2015_ND_CP": {"18": 4.0, "38": 2.0}},
    "civil_status_birth": {"60_2014_QH13": {"14": 7.0, "15": 3.0, "16": 2.0}, "123_2015_ND_CP": {"15": 3.0}},
    "civil_status_parent_child": {"60_2014_QH13": {"24": 7.0, "25": 3.0}, "123_2015_ND_CP": {"19": 3.0}},
    "adoption_conditions": {"52_2010_QH12": {"14": 8.0, "8": 5.0}},
    "adoption_procedure": {"52_2010_QH12": {"17": 6.0, "21": 5.0, "24": 5.0, "25": 5.0}, "19_2011_ND_CP": {"2": 4.0, "9": 4.0, "42": 4.0}, "24_2019_ND_CP": {"1": 4.0}},
    "domestic_violence_core": {"13_2022_QH15": {"2": 7.0, "3": 7.0, "5": 6.0, "8": 5.0, "19": 5.0, "25": 4.0}, "76_2023_ND_CP": {"1": 3.0, "2": 4.0, "3": 4.0}},
    "sanction_family": {"82_2020_ND_CP": {"59": 7.0, "60": 5.0, "61": 5.0, "62": 7.0}, "117_2024_ND_CP": {"1": 6.0}},
    "civil_procedure_family": {"92_2015_QH13": {"28": 8.0, "29": 8.0, "35": 6.0, "39": 7.0, "68": 4.0, "70": 4.0, "91": 6.0, "397": 6.0}},
    "civil_law_person": {"91_2015_QH13": {"16": 7.0, "21": 7.0, "22": 7.0, "26": 4.0, "27": 4.0, "28": 4.0, "41": 6.0}},
    "children_rights": {"102_2016_QH13": {"1": 4.0, "4": 5.0, "6": 6.0, "22": 6.0}, CORE_DOC_ID: {"69": 2.0}},
}

ROLE_HINTS: dict[str, list[str]] = {
    "procedure": ["thủ tục", "tòa án", "thẩm quyền", "khởi kiện", "xét đơn", "đương sự", "chứng cứ", "tố tụng", "hòa giải"],
    "sanction": ["xử phạt", "vi phạm hành chính", "phạt tiền", "tảo hôn", "tổ chức tảo hôn", "chế độ một vợ một chồng"],
    "supporting_property": ["năng lực hành vi", "năng lực pháp luật", "giao dịch dân sự", "giám hộ", "đại diện", "thừa kế", "họ tên", "nơi cư trú"],
    "supporting": ["hộ tịch", "khai sinh", "đăng ký kết hôn", "nhận cha", "nhận mẹ", "nhận con", "nuôi con nuôi", "trẻ em", "bạo lực gia đình"],
    "core": ["hôn nhân", "gia đình", "ly hôn", "vợ chồng", "quyền nuôi con", "cấp dưỡng", "tài sản chung", "tài sản riêng"],
}

NOISE_DOCS_WHEN_NOT_ASKED: dict[str, list[str]] = {
    "82_2020_ND_CP": ["xử phạt", "vi phạm", "phạt tiền", "tảo hôn", "hành chính"],
    "117_2024_ND_CP": ["xử phạt", "sửa đổi", "bổ sung", "nghị định 82", "82/2020"],
    "123_2015_ND_CP": ["hộ tịch", "ghi chú", "khai sinh", "đăng ký", "tình trạng hôn nhân", "có yếu tố nước ngoài"],
    "91_2015_QH13": ["dân sự", "năng lực", "giao dịch", "giám hộ", "thừa kế", "họ tên", "nơi cư trú"],
    "92_2015_QH13": ["tố tụng", "thủ tục", "tòa án", "thẩm quyền", "chứng cứ", "hòa giải", "công nhận thuận tình"],
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
    """Return the safe default route for strategy='auto'.

    Dense is the safest non-relation route on the current expanded family-law
    corpus. Hybrid_rerank remains available as an explicit strategy.
    """
    return "graph" if is_relation_query(query) else "dense"


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

def is_lottery_property_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in LOTTERY_PROPERTY_KEYWORDS)

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



def _contains_any(q: str, terms: Iterable[str]) -> bool:
    return any(term in q for term in terms)


def detect_legal_topic(query: str) -> str | None:
    """Detect a reusable legal intent/topic from a Vietnamese family-law query.

    The order is deliberate: specific child-custody/support/property intents must
    be detected before broad "ly hôn" or "tòa án" intents.
    """
    q = normalize_text(query)
    if not q:
        return None

    # Hôn nhân & ly hôn core
    if "thuận tình ly hôn" in q:
        return "voluntary_divorce"
    if "đơn phương ly hôn" in q or "ly hôn theo yêu cầu của một bên" in q or "yêu cầu của một bên" in q:
        return "unilateral_divorce"
    if "ai có quyền" in q and "ly hôn" in q:
        return "divorce_right"
    if "điều kiện kết hôn" in q:
        return "marriage_conditions"
    if "hành vi" in q and "bị cấm" in q and ("hôn nhân" in q or "gia đình" in q):
        return "prohibited_marriage_family"
    if "hủy" in q and "kết hôn trái pháp luật" in q:
        return "illegal_marriage_cancel"

    # Custody / child support. These are intentionally not one-off rules.
    if "thay đổi người trực tiếp nuôi con" in q:
        return "custody_change"
    if "thăm nom" in q or "cản trở" in q:
        return "custody_visitation"
    if "không trực tiếp nuôi con" in q and ("quyền" in q or "nghĩa vụ" in q):
        return "custody_obligation"
    if "cấp dưỡng" in q:
        return "child_support"
    if (
        "ly hôn" in q
        and _contains_any(
            q,
            [
                "quyền nuôi con",
                "giành quyền nuôi con",
                "trực tiếp nuôi con",
                "giao con",
                "chăm sóc con",
                "nuôi dưỡng con",
                "giáo dục con",
                "con dưới 36 tháng",
                "con từ đủ 7 tuổi",
                "nguyện vọng của con",
            ],
        )
    ):
        return "custody_initial_decision"

    # Property intents.
    if "tài sản chung" in q and "thời kỳ hôn nhân" in q and ("chia" in q or "thỏa thuận" in q):
        return "property_agreement"
    if "tài sản chung" in q and "vợ chồng" in q:
        return "common_property"
    if "tài sản riêng" in q:
        return "private_property"
    if "chia tài sản" in q and "ly hôn" in q:
        return "property_division_divorce"
    if "nguyên tắc chia tài sản" in q:
        return "property_division_divorce"

    # Civil status / adoption / children / domestic violence.
    if "đăng ký kết hôn" in q or "giấy xác nhận tình trạng hôn nhân" in q or "ghi chú ly hôn" in q:
        return "civil_status_marriage_registration"
    if "khai sinh" in q:
        return "civil_status_birth"
    if "nhận cha" in q or "nhận mẹ" in q or "nhận con" in q:
        return "civil_status_parent_child"
    if "nuôi con nuôi" in q or "con nuôi" in q or "cha mẹ nuôi" in q:
        if _contains_any(q, ["điều kiện", "người nhận", "người được nhận"]):
            return "adoption_conditions"
        return "adoption_procedure"
    if "bạo lực gia đình" in q or "cấm tiếp xúc" in q:
        return "domestic_violence_core"
    if "trẻ em" in q or "quyền trẻ em" in q or "sống chung với cha mẹ" in q:
        return "children_rights"

    # Sanction / civil procedure / civil law support.
    if _contains_any(q, ["xử phạt", "vi phạm hành chính", "phạt tiền", "tảo hôn", "tổ chức tảo hôn"]):
        return "sanction_family"
    if _contains_any(q, ["thẩm quyền", "tố tụng", "khởi kiện", "đương sự", "chứng cứ", "nghĩa vụ chứng minh"]):
        return "civil_procedure_family"
    if _contains_any(q, ["năng lực hành vi", "năng lực pháp luật", "giao dịch dân sự", "mất năng lực", "họ, tên", "nơi cư trú"]):
        return "civil_law_person"

    return None


def _article_boost_for_intent(intent: str | None, meta: dict) -> float:
    if not intent:
        return 0.0
    doc_id = str(meta.get("doc_id", "") or "").strip()
    article = str(meta.get("article", "") or "").strip().lower()
    if not doc_id:
        return 0.0

    doc_map = INTENT_ARTICLE_BOOSTS.get(intent, {})
    article_map = doc_map.get(doc_id)
    if not article_map:
        return 0.0

    score = 2.0  # doc-level boost when the doc is a known good authority for this intent.
    if article in {str(k).lower() for k in article_map.keys()}:
        for art, boost in article_map.items():
            if article == str(art).lower():
                score += float(boost)
                break
    return score


def _corpus_role_score(query: str, meta: dict) -> float:
    q = normalize_text(query)
    role = normalize_text(str(meta.get("corpus_role", "") or ""))
    doc_id = str(meta.get("doc_id", "") or "").strip()
    title = normalize_text(str(meta.get("title", "") or ""))

    score = 0.0
    for expected_role, hints in ROLE_HINTS.items():
        if role == expected_role and _contains_any(q, hints):
            score += 1.2

    # Core family-law statute should remain a strong default for core HN&GĐ queries.
    if doc_id == CORE_DOC_ID and _contains_any(q, ROLE_HINTS["core"]):
        score += 1.2

    # Title/domain-level support for documents that may not have corpus_role in older indexes.
    if "hộ tịch" in q and ("hộ tịch" in title or doc_id in DOC_TOPIC_HINTS["civil_status"]):
        score += 1.1
    if "nuôi con nuôi" in q and ("nuôi con nuôi" in title or doc_id in DOC_TOPIC_HINTS["adoption"]):
        score += 1.1
    if "bạo lực gia đình" in q and ("bạo lực gia đình" in title or doc_id in DOC_TOPIC_HINTS["domestic_violence"]):
        score += 1.1

    return score


def _noise_penalty(query: str, meta: dict) -> float:
    q = normalize_text(query)
    doc_id = str(meta.get("doc_id", "") or "").strip()
    if not doc_id:
        return 0.0

    required_terms = NOISE_DOCS_WHEN_NOT_ASKED.get(doc_id)
    if required_terms and not _contains_any(q, required_terms):
        # Keep penalties moderate: a clearly matching article can still win.
        return -1.1

    # Avoid procedure/sanction/property support docs stealing pure core-law answers.
    role = normalize_text(str(meta.get("corpus_role", "") or ""))
    core_query = _contains_any(q, ["ly hôn", "quyền nuôi con", "cấp dưỡng", "tài sản chung", "tài sản riêng"])
    if core_query and role in {"procedure", "sanction", "supporting_property"}:
        if not _contains_any(q, ROLE_HINTS.get(role, [])):
            return -0.8

    return 0.0


def legal_intent_score(query: str, doc: Document | dict) -> float:
    """Reusable intent-aware score for legal reranking and hybrid fusion."""
    meta = doc if isinstance(doc, dict) else (doc.metadata or {})
    intent = detect_legal_topic(query)
    score = 0.0
    score += _article_boost_for_intent(intent, meta)
    score += _corpus_role_score(query, meta)
    score += _noise_penalty(query, meta)
    return score


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
    score += legal_intent_score(query, doc)

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
