from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.tools import tool

from configs.setting import config
from src.indexing.bm25_index import load_bm25_index
from src.indexing.chroma_store import get_store
from src.retrieval.dense import dense_search
from src.retrieval.graph import build_graph, graph_search, is_relation_query
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank


@lru_cache(maxsize=1)
def _get_store():
    return get_store()


@lru_cache(maxsize=1)
def _get_bm25():
    return load_bm25_index()


@lru_cache(maxsize=1)
def _get_graph():
    cache_path = Path("data/graph/relationships.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        relationships = pickle.loads(cache_path.read_bytes())
    else:
        raw = load_dataset(
            getattr(config.dataset, "name", "th1nhng0/vietnamese-legal-documents"),
            name=getattr(config.dataset, "relationships_config", "relationships"),
            split=getattr(config.dataset, "split", "data"),
        )
        relationships = []
        for row in raw:
            src = str(row.get("doc_id", "")).strip()
            dst = str(row.get("other_doc_id", "")).strip()
            rel = str(row.get("relationship", "")).strip()
            if src and dst:
                relationships.append({"doc_id": src, "other_doc_id": dst, "relationship": rel})
        cache_path.write_bytes(pickle.dumps(relationships))
    return build_graph(relationships)


def _doc_to_dict(doc: Document) -> dict:
    return {"content": doc.page_content, "metadata": doc.metadata or {}}


def format_docs_for_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        citation = " | ".join(x for x in [
            str(meta.get("title", "")),
            f"Số hiệu: {meta.get('so_ky_hieu', '')}" if meta.get("so_ky_hieu") else "",
            f"Điều: {meta.get('article', '')}" if meta.get("article") else "",
            f"Hiệu lực: {meta.get('tinh_trang_hieu_luc', '')}" if meta.get("tinh_trang_hieu_luc") else "",
        ] if x)
        graph_path = meta.get("graph_path")
        graph_note = f"\nQuan hệ đồ thị: {graph_path}" if graph_path else ""
        blocks.append(f"[Nguồn {i}] {citation}{graph_note}\n{doc.page_content[:3500]}")
    return "\n\n".join(blocks)


def retrieve_documents(query: str, k: int = 5, strategy: str = "auto") -> list[Document]:
    store = _get_store()
    bm25 = _get_bm25()
    if strategy == "dense":
        return dense_search(store, query, k=k)
    if strategy in {"hybrid", "hybrid_rrf"}:
        return hybrid_search(store, bm25, query, k=k)
    if strategy == "hybrid_rerank":
        candidates = hybrid_search(store, bm25, query, k=max(k * 8, 40))
        return rerank(query, candidates, k=k, force=True)
    if strategy == "graph":
        return graph_search(store, bm25, _get_graph(), query, k=k, max_hops=getattr(config.retrieval, "graph_max_hops", 2))
    if strategy == "auto":
        if is_relation_query(query):
            return graph_search(store, bm25, _get_graph(), query, k=k, max_hops=getattr(config.retrieval, "graph_max_hops", 2))
        candidates = hybrid_search(store, bm25, query, k=max(k * 6, 30))
        return rerank(query, candidates, k=k)
    raise ValueError(f"Unknown strategy: {strategy}")


@tool
def dense_search_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Semantic vector search for general Vietnamese legal questions."""
    return [_doc_to_dict(d) for d in retrieve_documents(query, k=k, strategy="dense")]


@tool
def hybrid_search_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Hybrid RRF search: dense vector + Vietnamese BM25."""
    return [_doc_to_dict(d) for d in retrieve_documents(query, k=k, strategy="hybrid_rerank")]


@tool
def graph_traverse_tool(query: str, k: int = config.retrieval.k, max_hops: int = getattr(config.retrieval, "graph_max_hops", 2)) -> list[dict]:
    """Graph-guided retrieval for amendments, replacements, validity, references, and multi-hop legal relations."""
    docs = graph_search(_get_store(), _get_bm25(), _get_graph(), query, k=k, max_hops=max_hops)
    return [_doc_to_dict(d) for d in docs]


@tool
def retrieve_auto_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Automatically route query to hybrid or graph-guided retrieval."""
    return [_doc_to_dict(d) for d in retrieve_documents(query, k=k, strategy="auto")]


@tool
def generate_answer_tool(query: str, context: str) -> str:
    """Build a strict grounded-answer prompt from retrieved legal context."""
    return f"""
Bạn là trợ lý pháp luật Việt Nam, chuyên về Luật Hôn nhân và gia đình.

NHIỆM VỤ:
Trả lời trực tiếp câu hỏi của người dùng dựa trên NGỮ CẢNH được cung cấp.

[CÂU HỎI]
{query}

[NGỮ CẢNH]
{context}

[QUY TẮC TRẢ LỜI BẮT BUỘC]
1. Không được lặp lại câu hỏi.
2. Câu đầu tiên phải là KẾT LUẬN trực tiếp cho người hỏi.
3. Nếu câu hỏi hỏi "ưu tiên ai", phải trả lời rõ: "không tự động ưu tiên ai", hoặc "ưu tiên bên nào nếu..."
4. Sau kết luận, giải thích ngắn gọn các tiêu chí pháp luật mà Tòa án xem xét.
5. Nêu căn cứ pháp lý nếu có trong ngữ cảnh: tên văn bản, số hiệu, điều/khoản.
6. Nếu ngữ cảnh không đủ căn cứ, nói rõ: "Ngữ cảnh chưa đủ để kết luận chắc chắn", rồi nêu hướng phân tích theo luật.
7. Không bịa điều luật, không thêm nguồn ngoài ngữ cảnh.

[ĐỊNH DẠNG ĐẦU RA]
Kết luận: ...
Căn cứ: ...
Áp dụng vào tình huống: ...
"""
