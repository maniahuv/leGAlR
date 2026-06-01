from __future__ import annotations

import pickle
import re
import time
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.tools import tool

from configs.setting import config
from src.indexing.bm25_index import load_bm25_index
from src.indexing.chroma_store import get_store
from src.retrieval.dense import dense_search
from src.retrieval.graph import build_graph, graph_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_store():
    return get_store()


@lru_cache(maxsize=1)
def _get_bm25():
    return load_bm25_index()


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _load_jsonl_relationships(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import json

    relationships: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            src = str(row.get("doc_id", "")).strip()
            dst = str(row.get("other_doc_id", "")).strip()
            rel = str(row.get("relationship", "")).strip()
            if src and dst:
                relationships.append({"doc_id": src, "other_doc_id": dst, "relationship": rel})
    return relationships


@lru_cache(maxsize=1)
def _get_graph():
    graph_cfg = getattr(config, "graph", None)
    cache_path = _repo_path(getattr(graph_cfg, "persist_path", "data/graph/family_law_relationships.pkl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        relationships = pickle.loads(cache_path.read_bytes())
        return build_graph(relationships)

    source = str(getattr(config.dataset, "source", "huggingface")).lower().strip()
    if source == "local_pdf":
        raw_path = _repo_path(getattr(config.dataset, "relationships_path", "data/raw/family_law/relationships.jsonl"))
        relationships = _load_jsonl_relationships(raw_path)
        cache_path.write_bytes(pickle.dumps(relationships))
        return build_graph(relationships)

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


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

_GRAPH_INTENT_PATTERNS: tuple[str, ...] = (
    # Quan hệ sửa đổi / bổ sung / thay thế / bãi bỏ giữa các văn bản.
    # Lưu ý: KHÔNG route sang Graph chỉ vì query hỏi "còn hiệu lực"/"hết hiệu lực".
    # Các câu hỏi hiệu lực đơn giản nên đi Hybrid_Rerank vì metadata v4 đã xử lý rất tốt.
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

    # Văn bản hướng dẫn / quy định chi tiết.
    r"\bvăn\s+bản\s+hướng\s+dẫn\b",
    r"\bnghị\s+định\s+hướng\s+dẫn\b",
    r"\bthông\s+tư\s+hướng\s+dẫn\b",
    r"\bhướng\s+dẫn\s+thi\s+hành\b",
    r"\bquy\s+định\s+chi\s+tiết\b",
    r"\bquy\s+định\s+chi\s+tiết\s+thi\s+hành\b",

    # Dẫn chiếu / tham chiếu / quan hệ rõ ràng giữa văn bản.
    r"\bdẫn\s+chiếu\b",
    r"\btham\s+chiếu\b",
    r"\bquan\s+hệ\s+giữa\b",
    r"\bvăn\s+bản\s+nào\s+liên\s+quan\b",
    r"\bvăn\s+bản\s+liên\s+quan\s+đến\b",
)


def is_relation_query(query: str) -> bool:
    """
    Route sang Graph chỉ khi câu hỏi thật sự cần quan hệ giữa các văn bản:
    sửa đổi, bổ sung, thay thế, bãi bỏ, văn bản hướng dẫn,
    quy định chi tiết, dẫn chiếu hoặc tham chiếu.

    Các câu hỏi hỏi riêng về tình trạng hiệu lực/còn hiệu lực/hết hiệu lực
    sẽ đi Hybrid_Rerank để dùng metadata v4, tránh Graph chiếm quá nhiều route.

    Không route sang Graph chỉ vì query có các từ rất rộng như:
    "luật", "nghị định", "thông tư", "điều", "khoản", "văn bản",
    "hôn nhân", "gia đình", "ly hôn", "quyền nuôi con", "cấp dưỡng".
    Những query này thường nên đi qua Hybrid_Rerank để giữ độ chính xác
    ở câu hỏi tình huống và câu hỏi theo Điều/Khoản.
    """
    q = (query or "").lower().strip()
    if not q:
        return False

    return any(re.search(pattern, q) for pattern in _GRAPH_INTENT_PATTERNS)


def _attach_route(docs: list[Document], route: str) -> list[Document]:
    """
    Gắn route vào metadata để benchmark có thể đếm route_counts.
    Hàm này không thay đổi nội dung page_content, chỉ bổ sung metadata["route"].
    """
    for doc in docs:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["route"] = route
    return docs


# ---------------------------------------------------------------------------
# Formatting and utility helpers
# ---------------------------------------------------------------------------

def _doc_to_dict(doc: Document) -> dict:
    return {"content": doc.page_content, "metadata": doc.metadata or {}}


def format_docs_for_context(docs: list[Document]) -> str:
    blocks = []
    max_chars = int(getattr(config.retrieval, "context_chars_per_doc", 2300))

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        citation = " | ".join(
            x
            for x in [
                str(meta.get("title", "")),
                f"Số hiệu: {meta.get('so_ky_hieu', '')}" if meta.get("so_ky_hieu") else "",
                f"Điều: {meta.get('article', '')}" if meta.get("article") else "",
                f"Khoản: {meta.get('clause', '')}" if meta.get("clause") else "",
                f"Hiệu lực: {meta.get('tinh_trang_hieu_luc', '')}"
                if meta.get("tinh_trang_hieu_luc")
                else "",
            ]
            if x
        )
        graph_path = meta.get("graph_path")
        graph_note = f"\nQuan hệ đồ thị: {graph_path}" if graph_path else ""
        blocks.append(
            f"[Nguồn {i}] {citation}{graph_note}\n{(doc.page_content or '')[:max_chars]}"
        )

    return "\n\n".join(blocks)


def _pool_size(k: int, accurate: bool = False) -> int:
    multiplier = int(getattr(config.retrieval, "pool_multiplier", 6))
    minimum = int(getattr(config.retrieval, "min_pool_size", 30))

    if accurate:
        multiplier = max(
            multiplier, int(getattr(config.retrieval, "accurate_pool_multiplier", 8))
        )
        minimum = max(minimum, int(getattr(config.retrieval, "accurate_min_pool_size", 40)))

    return max(k * multiplier, minimum)


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve_documents(query: str, k: int = 5, strategy: str = "auto") -> list[Document]:
    """
    Retrieval entry point.

    Strategy:
    - dense: vector search baseline.
    - hybrid / hybrid_rrf: dense + BM25 + legal metadata signals.
    - hybrid_rerank: hybrid candidate pool + legal-aware reranker.
    - graph: graph-guided retrieval for legal document relationships.
    - auto:
        + graph only for relation/effectiveness queries.
        + hybrid_rerank for normal legal QA, title/metadata, exact article,
          and family-law semantic questions.
    """
    t0 = time.perf_counter()
    store = _get_store()
    bm25 = _get_bm25()
    route = strategy

    if strategy == "dense":
        docs = dense_search(store, query, k=k)
        route = "dense"

    elif strategy in {"hybrid", "hybrid_rrf"}:
        docs = hybrid_search(store, bm25, query, k=k)
        route = "hybrid"

    elif strategy == "hybrid_rerank":
        candidates = hybrid_search(store, bm25, query, k=_pool_size(k, accurate=True))
        docs = rerank(query, candidates, k=k, force=True)
        route = "hybrid_rerank"

    elif strategy == "graph":
        docs = graph_search(
            store,
            bm25,
            _get_graph(),
            query,
            k=k,
            max_hops=getattr(config.retrieval, "graph_max_hops", 1),
        )
        route = "graph"

    elif strategy == "auto":
        if is_relation_query(query):
            route = "graph"
            docs = graph_search(
                store,
                bm25,
                _get_graph(),
                query,
                k=k,
                max_hops=getattr(config.retrieval, "graph_max_hops", 1),
            )
        else:
            route = "hybrid_rerank"
            candidates = hybrid_search(
                store,
                bm25,
                query,
                k=_pool_size(k, accurate=True),
            )
            docs = rerank(query, candidates, k=k, force=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    docs = _attach_route(docs, route)

    if bool(getattr(config.retrieval, "log_retrieval", False)):
        latency_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[retrieval] strategy={strategy} route={route} k={k} "
            f"docs={len(docs)} latency={latency_ms:.1f}ms query={query[:100]}"
        )

    return docs


# ---------------------------------------------------------------------------
# LangChain / LangGraph tools
# ---------------------------------------------------------------------------

@tool
def dense_search_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Semantic vector search for general Vietnamese legal questions."""
    return [_doc_to_dict(d) for d in retrieve_documents(query, k=k, strategy="dense")]


@tool
def hybrid_search_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Legal-aware hybrid search: dense vector + Vietnamese BM25 + legal rerank."""
    return [_doc_to_dict(d) for d in retrieve_documents(query, k=k, strategy="hybrid_rerank")]


@tool
def graph_traverse_tool(
    query: str,
    k: int = config.retrieval.k,
    max_hops: int = getattr(config.retrieval, "graph_max_hops", 1),
) -> list[dict]:
    """Graph-guided retrieval for amendments, replacements, validity, references, and multi-hop legal relations."""
    docs = graph_search(_get_store(), _get_bm25(), _get_graph(), query, k=k, max_hops=max_hops)
    docs = _attach_route(docs, "graph")
    return [_doc_to_dict(d) for d in docs]


@tool
def retrieve_auto_tool(query: str, k: int = config.retrieval.k) -> list[dict]:
    """Automatically route query to legal-aware hybrid rerank or graph-guided retrieval."""
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
3. Nếu câu hỏi hỏi "ưu tiên ai", phải trả lời rõ: "không tự động ưu tiên ai", hoặc "ưu tiên bên nào nếu...".
4. Ưu tiên văn bản còn hiệu lực; nếu nguồn có dấu hiệu hết hiệu lực thì phải cảnh báo.
5. Sau kết luận, giải thích ngắn gọn các tiêu chí pháp luật mà Tòa án/cơ quan có thẩm quyền xem xét.
6. Nêu căn cứ pháp lý nếu có trong ngữ cảnh: tên văn bản, số hiệu, điều/khoản.
7. Nếu ngữ cảnh không đủ căn cứ, nói rõ: "Ngữ cảnh chưa đủ để kết luận chắc chắn", rồi nêu hướng phân tích theo luật.
8. Không bịa điều luật, không thêm nguồn ngoài ngữ cảnh.

[ĐỊNH DẠNG ĐẦU RA]
Kết luận: ...
Căn cứ: ...
Áp dụng vào tình huống: ...
"""
