import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset
from langchain_core.tools import tool
from langchain_core.documents import Document

from configs.setting import config

from src.indexing.chroma_store import get_store
from src.indexing.bm25_index import load_bm25_index

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.graph import build_graph, graph_search


@lru_cache(maxsize=1)
def _get_store():
    return get_store()


@lru_cache(maxsize=1)
def _get_bm25():
    return load_bm25_index()


@lru_cache(maxsize=1)
def _get_graph():
    path = Path("data/graph/relationships.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "rb") as f:
            relationships = pickle.load(f)
    else:
        relationships = load_dataset(
            "th1nhng0/vietnamese-legal-documents",
            name="relationships",
            split="data",
        )

        relationships = list(relationships)

        with open(path, "wb") as f:
            pickle.dump(relationships, f)

    return build_graph(relationships)


def _doc_to_dict(doc: Document) -> dict:
    return {
        "content": doc.page_content,
        "metadata": doc.metadata or {},
    }


@tool
def dense_search_tool(
    query: str,
    k: int = config.retrieval.k,
) -> list[dict]:
    """
    Retrieve legal documents using dense vector search.
    Use this for general semantic legal questions.
    """
    docs = dense_search(
        _get_store(),
        query,
        k=k,
    )

    return [_doc_to_dict(d) for d in docs]


@tool
def hybrid_search_tool(
    query: str,
    k: int = config.retrieval.k,
) -> list[dict]:
    """
    Retrieve legal documents using hybrid search: BM25 + vector search.
    Use this for exact legal keywords, article numbers, document numbers,
    dates, and legal effectiveness queries.
    """
    docs = hybrid_search(
        _get_store(),
        _get_bm25(),
        query,
        k=k * 2,
    )

    docs = rerank(query, docs, k=k)

    return [_doc_to_dict(d) for d in docs]


@tool
def graph_traverse_tool(
    query: str,
    k: int = config.retrieval.k,
    initial_k: int = 3,
    max_hops: int = getattr(config.retrieval, "graph_max_hops", 2),
) -> list[dict]:
    """
    Retrieve documents using graph-guided multi-hop retrieval.
    Seed with dense search, then expand via relationship graph edges.
    Use this for multi-hop questions about document history,
    amendments, replacements, abolitions, references, or legal hierarchy.
    """
    try:
        docs = graph_search(
            _get_store(),
            _get_graph(),
            query,
            k=k,
            initial_k=initial_k,
            max_hops=max_hops,
        )

        return [_doc_to_dict(d) for d in docs]

    except Exception as e:
        return [{"error": str(e)}]


@tool
def generate_answer_tool(
    query: str,
    context: str,
) -> str:
    """
    Generate a final legal answer from retrieved legal context.
    Use this after retrieval tools.
    """
    return (
        "Hãy trả lời câu hỏi pháp luật Việt Nam dựa trên ngữ cảnh sau.\n\n"
        f"Câu hỏi: {query}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        "Yêu cầu:\n"
        "- Trả lời rõ ràng, dễ hiểu.\n"
        "- Nếu văn bản đã hết hiệu lực, phải nói rõ.\n"
        "- Không bịa thông tin ngoài ngữ cảnh."
    )