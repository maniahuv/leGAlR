import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from configs.setting import config


def build_bm25_index(docs: list[Document]) -> BM25Retriever:
    """
    Build BM25 index từ danh sách Document.
    """
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = config.retrieval.k
    return retriever


def save_bm25_index(retriever: BM25Retriever):
    """
    Lưu BM25 index xuống file.
    """
    path = Path(config.bm25.persist_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(retriever, f)


def load_bm25_index() -> BM25Retriever:
    """
    Load BM25 index từ file.
    """
    path = Path(config.bm25.persist_path)

    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found: {path}")

    with open(path, "rb") as f:
        retriever = pickle.load(f)

    return retriever