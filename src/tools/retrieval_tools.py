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
    """
    Nạp và cache Chroma vector store.
    """
    return get_store()


@lru_cache(maxsize=1)
def _get_bm25():
    """
    Nạp và cache chỉ mục từ khóa BM25.
    """
    return load_bm25_index()


@lru_cache(maxsize=1)
def _get_graph():
    """
    Nạp, chuẩn hóa dữ liệu quan hệ văn bản pháp luật và cache đồ thị mạng lưới.
    """
    path = Path("data/graph/relationships.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, "rb") as f:
            relationships = pickle.load(f)
    else:
        print("⏳ Đang tải tập dữ liệu quan hệ văn bản gốc từ HuggingFace...")
        raw_relationships = load_dataset(
            "th1nhng0/vietnamese-legal-documents",
            name="relationships",
            split="data",
        )
        
        # 🎯 BƯỚC CHUẨN HÓA CỐT LÕI: Ép kiểu toàn bộ ID thành chuỗi (str) và loại bỏ khoảng trắng
        relationships = []
        for row in raw_relationships:
            src = str(row.get("doc_id", "")).strip()
            dst = str(row.get("other_doc_id", "")).strip()
            rel_type = str(row.get("relationship", "")).strip()
            if src and dst:
                relationships.append({
                    "doc_id": src,
                    "other_doc_id": dst,
                    "relationship": rel_type
                })

        with open(path, "wb") as f:
            pickle.dump(relationships, f)

    return build_graph(relationships)


def _doc_to_dict(doc: Document) -> dict:
    """
    Hỗ trợ chuyển đổi cấu trúc đối tượng Document sang Dictionary thuần túy.
    """
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
    # Lấy pool ứng viên rộng hơn (k * 4) để tầng Rerank làm việc hiệu quả
    docs = hybrid_search(
        _get_store(),
        _get_bm25(),
        query,
        k=k * 4,
    )

    # Tiến hành xếp hạng lại bằng cơ chế Bi-gram Jaccard Similarity đã tối ưu
    docs = rerank(query, docs, k=k)

    return [_doc_to_dict(d) for d in docs]


@tool
def graph_traverse_tool(
    query: str,
    k: int = config.retrieval.k,
    max_hops: int = getattr(config.retrieval, "graph_max_hops", 2),
) -> list[dict]:
    """
    Retrieve documents using graph-guided multi-hop retrieval.
    Seed with dense search, then expand via relationship graph edges.
    Use this for multi-hop questions about document history,
    amendments, replacements, abolitions, references, or legal hierarchy.
    """
    try:
        graph_obj = _get_graph()
        docs = graph_search(
            _get_store(),
            graph_obj,
            query,
            k=k,
            max_hops=max_hops,
        )
        return [_doc_to_dict(d) for d in docs]

    except Exception as e:
        return [{"error": f"Lỗi thực thi đồ thị tại tool: {str(e)}"}]


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
        "- Trả lời rõ ràng, dễ hiểu, bám sát căn cứ pháp lý trong ngữ cảnh.\n"
        "- Nếu văn bản đã hết hiệu lực hoặc bị thay thế, phải nêu rõ ràng cho người dùng.\n"
        "- Tuyệt đối không tự bịa đặt hay suy diễn thông tin nằm ngoài ngữ cảnh cung cấp."
    )