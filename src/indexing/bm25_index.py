from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from configs.setting import config

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None


LEGAL_STOPWORDS = {
    "là", "gì", "của", "được", "theo", "về", "và", "trong", "các", "những", "cho",
    "đến", "nào", "đã", "đang", "sẽ", "có", "thì", "mà", "một", "như", "khi", "nếu",
}


def tokenize_vi(text: str) -> list[str]:
    text = (text or "").lower()
    if word_tokenize is not None:
        text = word_tokenize(text, format="text")
    tokens = re.findall(r"[\w/\.-]+", text, flags=re.UNICODE)
    return [t for t in tokens if t and t not in LEGAL_STOPWORDS]


class VietnameseBM25Retriever:
    def __init__(self, documents: list[Document], k: int = 5):
        self.documents = documents
        self.k = k
        self.tokenized_docs = [tokenize_vi(doc.page_content + " " + _metadata_text(doc)) for doc in documents]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None
            self.doc_freq = self._build_doc_freq(self.tokenized_docs)

    @staticmethod
    def _build_doc_freq(tokenized_docs: list[list[str]]) -> dict[str, int]:
        df: dict[str, int] = {}
        for toks in tokenized_docs:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        return df

    def get_scores(self, query: str):
        q_tokens = tokenize_vi(query)
        if self.bm25 is not None:
            return self.bm25.get_scores(q_tokens)
        # Fallback TF-IDF overlap nếu chưa cài rank_bm25.
        scores = []
        n_docs = max(len(self.tokenized_docs), 1)
        for toks in self.tokenized_docs:
            counts = {t: toks.count(t) for t in set(toks)}
            score = 0.0
            for q in q_tokens:
                if q in counts:
                    idf = math.log((n_docs + 1) / (self.doc_freq.get(q, 0) + 1)) + 1
                    score += counts[q] * idf
            scores.append(score)
        return scores

    def invoke(self, query: str, **kwargs) -> list[Document]:
        scores = list(self.get_scores(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.documents[i] for i in top if scores[i] > 0 or top]

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


def _metadata_text(doc: Document) -> str:
    meta = doc.metadata or {}
    return " ".join(str(meta.get(k, "")) for k in [
        "title", "so_ky_hieu", "loai_van_ban", "linh_vuc", "nganh", "tinh_trang_hieu_luc",
        "article", "clause",
    ])


def build_bm25_index(docs: list[Document]) -> VietnameseBM25Retriever:
    return VietnameseBM25Retriever(docs, k=int(config.retrieval.k))


def save_bm25_index(retriever: VietnameseBM25Retriever):
    path = Path(config.bm25.persist_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(retriever, f)


def load_bm25_index() -> VietnameseBM25Retriever:
    path = Path(config.bm25.persist_path)
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}. Run: python scripts/ingest.py")
    with open(path, "rb") as f:
        return pickle.load(f)
