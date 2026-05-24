import time
from collections import defaultdict
from src.indexing.chroma_store import get_store
from src.indexing.bm25_index import load_bm25_index
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.evaluation.metrics import evaluate_retrieval_case, average_metrics
from src.retrieval.graph import graph_search

def _doc_id(doc) -> str:
    return str((doc.metadata or {}).get("doc_id", "")).strip()

def run_retrieval_benchmark(
    test_cases: list[dict],
    strategy: str = "hybrid",
    k: int = 5,
) -> dict:
    """
    Hàm chạy thực nghiệm (Benchmark) đánh giá hiệu năng các chiến lược truy hồi.
    """
    store = get_store()
    bm25 = load_bm25_index()
    
    graph_obj = None
    if strategy == "graph":
        try:
            from src.tools.retrieval_tools import _get_graph
            graph_obj = _get_graph()
            if graph_obj is None or len(graph_obj) == 0:
                print("⚠️ Cảnh báo: Đồ thị trống hoặc chưa được tải thành công!")
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi khởi tạo đồ thị trong Benchmark: {e}")

    case_results = []
    scenario_metrics = defaultdict(list)

    for case in test_cases:
        query = case["query"]
        scenario = case.get("scenario", "single-hop-semantic")
        relevant_ids = [str(x).strip() for x in case.get("relevant_ids", [])]

        # Thực thi các chiến lược truy hồi
        if strategy == "dense":
            docs = dense_search(store, query, k=k)
            
        elif strategy == "hybrid":
            docs = hybrid_search(store, bm25, query, k=k)
            
        elif strategy == "hybrid_rerank":
            # Nới rộng pool ứng viên thô (k * 4) để Reranker có không gian xếp hạng lại tốt hơn
            candidates = hybrid_search(store, bm25, query, k=k * 4)
            docs = rerank(query, candidates, k=k)
            
        elif strategy == "graph":
            if graph_obj is not None:
                try:
                    # Gọi trực tiếp thuật toán loang đồ thị nâng cao
                    docs = graph_search(store, graph_obj, query, k=k, max_hops=2)
                except Exception as e:
                    print(f"❌ Lỗi thực thi graph_search cho câu hỏi '{query}': {e}. Fallback về Hybrid.")
                    docs = hybrid_search(store, bm25, query, k=k)
            else:
                docs = hybrid_search(store, bm25, query, k=k)
        else:
            docs = []

        # Lấy danh sách ID độc nhất từ top-k tài liệu được truy hồi
        retrieved_ids = []
        for doc in docs:
            d_id = _doc_id(doc)
            if d_id and d_id not in retrieved_ids:
                retrieved_ids.append(d_id)
        
        # Cắt chính xác lấy top-k ID sau khi loại trùng
        retrieved_ids = retrieved_ids[:k]

        # Tính toán toàn bộ các chỉ số (Hit@K, Precision@K, Recall@K, MRR)
        metrics = evaluate_retrieval_case(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        scenario_metrics[scenario].append(metrics)
        scenario_metrics["global_overall"].append(metrics)

        case_results.append(
            {
                "query": query,
                "scenario": scenario,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "metrics": metrics,
            }
        )

    # Tổng hợp kết quả trung bình theo từng bài toán / kịch bản câu hỏi
    summary_by_scenario = {}
    for sc, metric_list in scenario_metrics.items():
        summary_by_scenario[sc] = average_metrics(metric_list)

    return {
        "strategy": strategy,
        "k": k,
        "summary": summary_by_scenario.get("global_overall", {}),
        "breakdown": summary_by_scenario,
        "cases": case_results,
    }