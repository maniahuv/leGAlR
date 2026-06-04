from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

from langchain_core.documents import Document

from configs.setting import config
from src.retrieval.legal_signals import LEGAL_STOPWORDS, legal_token_expansion, metadata_text

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover
    word_tokenize = None


def tokenize_vi(text: str) -> list[str]:
    text = (text or "").lower()
    if word_tokenize is not None:
        text = word_tokenize(text, format="text")
    tokens = re.findall(r"[\w/\.\-]+", text, flags=re.UNICODE)
    tokens = [t for t in tokens if t and t not in LEGAL_STOPWORDS]
    tokens.extend(legal_token_expansion(text))
    return tokens


class VietnameseBM25Retriever:
    def __init__(self, documents: list[Document], k: int = 5):
        self.documents = documents
        self.k = k
        self.tokenized_docs = [tokenize_vi(_document_search_text(doc)) for doc in documents]
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
        if not scores:
            return []
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        # Important: do not return zero-score BM25 hits. Dense retrieval remains the fallback in hybrid_search.
        return [self.documents[i] for i in top if scores[i] > 0]

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


def _document_search_text(doc: Document) -> str:
    return f"{doc.page_content or ''} {metadata_text(doc)}"


def _metadata_text(doc: Document) -> str:
    # Backward-compatible private helper used by older pickles/tests.
    return metadata_text(doc)


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
        retriever = pickle.load(f)
    return retriever
