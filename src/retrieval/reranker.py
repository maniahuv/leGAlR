from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None

from configs.setting import config

LEGAL_STOPWORDS = {
    "là", "gì", "của", "được", "theo", "về", "và", "trong", "các", "những", "cho",
    "đến", "nào", "đã", "đang", "sẽ", "có", "thì", "mà", "một", "như", "khi", "nếu",
}

EXACT_QUERY_PATTERNS = [
    r"\bđiều\s+\d+", r"\bkhoản\s+\d+", r"\bđiểm\s+[a-z]", r"\d+\/\d{4}\/[A-ZĐ\-]+",
    r"nghị định", r"thông tư", r"luật", r"hiệu lực", r"hết hiệu lực", r"sửa đổi", r"thay thế",
]


def should_rerank(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(p, q, flags=re.IGNORECASE) for p in EXACT_QUERY_PATTERNS)


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    if word_tokenize is not None:
        text = word_tokenize(text, format="text")
    tokens = re.findall(r"[\w/\.-]+", text, flags=re.UNICODE)
    return {t for t in tokens if t not in LEGAL_STOPWORDS}


def _extract_identifiers(text: str) -> set[str]:
    t = (text or "").lower()
    ids = set()
    for m in re.findall(r"\bđiều\s+(\d+[a-zA-Z]?)", t):
        ids.add(f"dieu_{m}")
    for m in re.findall(r"\bkhoản\s+(\d+)", t):
        ids.add(f"khoan_{m}")
    for m in re.findall(r"\bđiểm\s+([a-z])", t):
        ids.add(f"diem_{m}")
    for m in re.findall(r"\b\d+\/\d{4}\/[\w\-Đđ]+", t, flags=re.UNICODE):
        ids.add(m.lower())
    return ids


def _metadata_text(doc: Document) -> str:
    meta = doc.metadata or {}
    return " ".join(str(meta.get(k, "")) for k in [
        "title", "so_ky_hieu", "loai_van_ban", "linh_vuc", "nganh", "tinh_trang_hieu_luc", "article", "clause",
    ])


def _rule_score(query: str, doc: Document) -> float:
    q = query or ""
    doc_text = (doc.page_content or "") + " " + _metadata_text(doc)
    q_tokens = _tokenize(q)
    d_tokens = _tokenize(doc_text)
    if not q_tokens:
        overlap = 0.0
    else:
        overlap = len(q_tokens & d_tokens) / (len(q_tokens) ** 0.5)

    q_ids = _extract_identifiers(q)
    d_ids = _extract_identifiers(doc_text)
    id_bonus = 0.0
    if q_ids:
        id_bonus += 0.65 * len(q_ids & d_ids)

    meta = doc.metadata or {}
    metadata_bonus = 0.0
    so = str(meta.get("so_ky_hieu", "")).lower()
    if so and so in q.lower():
        metadata_bonus += 0.55
    title_tokens = _tokenize(str(meta.get("title", "")))
    if title_tokens & q_tokens:
        metadata_bonus += 0.12
    if str(meta.get("article", "")) and f"điều {meta.get('article')}" in q.lower():
        metadata_bonus += 0.35

    status = str(meta.get("tinh_trang_hieu_luc", "")).lower()
    status_bonus = 0.03 if "còn hiệu lực" in status else (-0.02 if "hết hiệu lực" in status else 0.0)
    graph_bonus = 0.12 / max(int(meta.get("graph_distance", 1) or 1), 1) if meta.get("graph_distance") else 0.0
    return overlap + id_bonus + metadata_bonus + status_bonus + graph_bonus


@lru_cache(maxsize=1)
def _cross_encoder():
    if not bool(getattr(config.retrieval, "use_cross_encoder", False)):
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(getattr(config.retrieval, "cross_encoder_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    except Exception as e:
        print(f"Không tải được CrossEncoder, fallback rule-based reranker: {e}")
        return None


def rerank(query: str, docs: list[Document], k: int = 5, force: bool = False) -> list[Document]:
    if not docs:
        return []
    if not force and not should_rerank(query):
        return docs[:k]

    ce = _cross_encoder()
    if ce is not None:
        pairs = [(query, (doc.page_content or "")[:3500]) for doc in docs]
        ce_scores = ce.predict(pairs)
        scored = [(float(s) + _rule_score(query, doc), doc) for s, doc in zip(ce_scores, docs)]
    else:
        scored = [(_rule_score(query, doc), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]
