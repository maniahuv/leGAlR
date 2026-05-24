import networkx as nx
from langchain_core.documents import Document
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search

def build_graph(relationships: list[dict]) -> nx.DiGraph:
    """
    Khởi tạo đồ thị có hướng biểu diễn mối quan hệ giữa các văn bản pháp luật.
    """
    graph = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", ""))
        dst = str(row.get("other_doc_id", ""))
        rel_type = row.get("relationship", "")
        if src and dst:
            graph.add_edge(src, dst, rel_type=rel_type)
    return graph

def graph_search(store, graph: nx.DiGraph, query: str, k: int = 5, initial_k: int = 3, max_hops: int = 2):
    """
    Thuật toán tìm kiếm nâng cao dẫn đường bằng Đồ thị quan hệ hiệu lực văn bản (Graph-guided RAG).
    """
    # Nới rộng không gian tìm kiếm thô để ngăn chặn hiện tượng nghẽn chunk
    POOL_K = k * 5
    
    # 1. Gọi trực tiếp mô hình hybrid_search để lấy tài liệu hạt nhân chuẩn xác (Seed Docs)
    # Để tránh bị ép chặt unique quá sớm, ta nạp đối tượng chỉ mục trực tiếp
    from src.indexing.bm25_index import load_bm25_index
    bm25 = load_bm25_index()
    
    seed_docs = hybrid_search(store, bm25, query, k=POOL_K)
    seed_ids = {str(d.metadata.get("doc_id", "")) for d in seed_docs if d.metadata.get("doc_id")}

    reachable, frontier = set(seed_ids), set(seed_ids)
    
    # 2. Thuật toán loang đồ thị tìm các văn bản sửa đổi, thay thế hoặc dẫn chiếu liên quan
    for _ in range(max_hops):
        nxt = set()
        for node in frontier:
            if node not in graph:
                continue
            nxt |= {str(nb) for _, nb, _ in graph.out_edges(node, data=True) if str(nb) not in reachable}
            nxt |= {str(nb) for nb, _, _ in graph.in_edges(node, data=True) if str(nb) not in reachable}
        reachable |= nxt
        frontier = nxt
        if not frontier:
            break
    
    extra_ids = list(reachable - seed_ids)
    extra_docs = []
    
    # Giới hạn lấy tối đa 30 ID loang gần nhất để kiểm soát nhiễu, tránh chèn ép vị trí
    if len(extra_ids) > 30:
        extra_ids = extra_ids[:30]

    # 3. Chia nhỏ danh sách IDs thành từng Batch tối đa 200 phần tử phòng vệ lỗi SQLite
    if extra_ids:
        BATCH_SIZE = 200
        for i in range(0, len(extra_ids), BATCH_SIZE):
            batch_ids = extra_ids[i:i + BATCH_SIZE]
            try:
                # Tìm các đoạn văn bản thuộc dải ID quan hệ
                batch_docs = dense_search(
                    store, 
                    query, 
                    k=5,  # Lấy lượng vừa đủ để không làm loãng khối kết quả
                    metadata_filter={"doc_id": {"$in": batch_ids}}
                )
                extra_docs.extend(batch_docs)
            except Exception:
                continue
    
    merged: list[Document] = []
    seen_văn_bản = set()

    # 4. ĐẢO NGƯỢC CHIẾN LƯỢC XẾP HẠNG: Đề cao tài liệu hạt nhân cốt lõi (Seed Docs) lên trước
    for doc in seed_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", ""))
        if doc_id and doc_id not in seen_văn_bản:
            seen_văn_bản.add(doc_id)
            merged.append(doc)

    # Sau đó mới chèn bổ sung các tài liệu bắc cầu quan hệ (Extra Docs) vào các vị trí trống còn lại
    for doc in extra_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", ""))
        if doc_id and doc_id not in seen_văn_bản:
            seen_văn_bản.add(doc_id)
            merged.append(doc)
            
    return merged[:k]