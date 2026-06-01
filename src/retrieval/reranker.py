from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.legal_signals import (
    extract_identifiers,
    metadata_signal_score,
    metadata_text,
    should_rerank,
    tokenize,
)


def _rule_score(query: str, doc: Document) -> float:
    q_tokens = tokenize(query)
    doc_text = f"{doc.page_content or ''} {metadata_text(doc)}"
    d_tokens = tokenize(doc_text)

    if not q_tokens:
        overlap = 0.0
    else:
        # Normalized lexical overlap. This rewards exact legal terms without overfavoring long queries.
        overlap = len(q_tokens & d_tokens) / (len(q_tokens) ** 0.5)

    q_ids = extract_identifiers(query)
    d_ids = extract_identifiers(doc_text)
    exact_id_bonus = 0.0
    if q_ids:
        exact_id_bonus += 0.70 * len(q_ids & d_ids)

    return overlap + exact_id_bonus + metadata_signal_score(query, doc)


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
        pairs = [(query, (doc.page_content or "")[: int(getattr(config.retrieval, "cross_encoder_max_chars", 3000))]) for doc in docs]
        ce_scores = ce.predict(pairs)
        scored = [(float(s) + _rule_score(query, doc), doc) for s, doc in zip(ce_scores, docs)]
    else:
        scored = [(_rule_score(query, doc), doc) for doc in docs]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]
