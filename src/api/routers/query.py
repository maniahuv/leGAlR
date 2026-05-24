import time
from src.retrieval.graph import graph_search

def _agentic_retrieve(question: str, k: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    q_lower =question.lower()

    if any(kw in q_lower for kw in
           ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu"]): # kiểu quan hệ giữa các văn bản
        docs = graph_search(_get_store(), _get_graph(), question, k=k)
    elif any(kw in q_lower for kw in
            ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm"]): # cần truy keyword chính xác theo năm, tháng, ngày
        docs = hybrid_search(_get_store(),_get_bm25(), question, k=k)
    else:
        candidates = hybrid_search(_get_store(), _get_bm25(), question, k =k*2)
        docs = rerank(question, candidates, k=k)
    
    latency_ms = (time.perf_counter() - t0) * 1000
    return [{"page_content": d.page_content,
             "metadata": d.metadata} for d in docs], latency_ms