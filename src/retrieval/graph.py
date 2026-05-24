import networkx as nx
from langchain_core.documents import Document
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search

def build_graph(relationships: list[dict]) -> nx.DiGraph:
    """
    Khởi tạo đồ thị có hướng biểu diễn mối quan hệ hiệu lực, sửa đổi, bổ sung giữa các văn bản pháp luật.
    """
    graph = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", "")).strip()
        dst = str(row.get("other_doc_id", "")).strip()
        rel_type = str(row.get("relationship", "")).strip()
        if src and dst:
            graph.add_edge(src, dst, rel_type=rel_type)
    return graph

def graph_search(store, graph: nx.DiGraph, query: str, k: int = 5, max_hops: int = 2):
    """
    Thuật toán tìm kiếm nâng cao dẫn đường bằng Đồ thị quan hệ hiệu lực văn bản (Graph-guided RAG).
    Sửa lỗi cú pháp toán tử $in tương thích sâu với cơ chế lọc của ChromaDB / LangChain Chroma.
    """
    # 1. Gọi mô hình hybrid_search để lấy các tài liệu hạt nhân chuẩn xác (Seed Docs) làm gốc loang
    from src.indexing.bm25_index import load_bm25_index
    bm25 = load_bm25_index()
    
    # Lấy pool ứng viên rộng để làm giàu không gian loang hạt nhân ban đầu
    seed_docs = hybrid_search(store, bm25, query, k=k * 3)
    seed_ids = {str(d.metadata.get("doc_id", "")).strip() for d in seed_docs if d.metadata.get("doc_id")}

    reachable = set(seed_ids)
    frontier = set(seed_ids)
    
    # 2. Thuật toán loang đồ thị tìm các văn bản sửa đổi, thay thế hoặc dẫn chiếu liên quan (BFS)
    for _ in range(max_hops):
        nxt = set()
        for node in frontier:
            if node not in graph:
                continue
            
            # Lấy láng giềng đi ra (nút bị node này trỏ tới)
            for u, v in graph.out_edges(node):
                v_str = str(v).strip()
                if v_str not in reachable:
                    nxt.add(v_str)
                    
            # Lấy láng giềng đi vào (nút trỏ tới node này)
            for v, u in graph.in_edges(node):
                v_str = str(v).strip()
                if v_str not in reachable:
                    nxt.add(v_str)
                    
        reachable |= nxt
        frontier = nxt
        if not frontier:
            break
    
    extra_ids = list(reachable - seed_ids)
    extra_docs = []
    
    # Giới hạn lấy tối đa 30 ID loang gần nhất để kiểm soát nhiễu
    if len(extra_ids) > 30:
        extra_ids = extra_ids[:30]

    # 3. TRUY VẤN ĐOẠN VĂN BẢN MỞ RỘNG (Sửa lỗi toán tử logic của ChromaDB)
    if extra_ids:
        # CÚ PHÁP CHUẨN ĐÚNG CỦA LANGCHAIN CHROMA: Sử dụng dict toán tử logic lồng nhau {"$in": [...]}
        # Ép kiểu chặt chẽ từng phần tử ID bên trong thành chuỗi thuần túy
        chroma_filter = {"doc_id": {"$in": [str(eid).strip() for eid in extra_ids]}}
        
        try:
            # Tìm các đoạn văn bản thuộc dải ID quan hệ mở rộng
            batch_docs = dense_search(
                store, 
                query, 
                k=15, # Nới rộng không gian ngữ cảnh bổ sung  
                metadata_filter=chroma_filter
            )
            extra_docs.extend(batch_docs)
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi thực thi metadata_filter trong ChromaDB: {e}")
    
    merged: list[Document] = []
    seen_chunks = set()

    # 4. CHIẾN LƯỢC XẾP HẠNG PHỐI HỢP: Đề cao tài liệu hạt nhân cốt lõi (Seed Docs) lên trước
    for doc in seed_docs:
        chunk_uid = f"{doc.metadata.get('doc_id')}_{doc.metadata.get('chunk_index', 0)}"
        if chunk_uid not in seen_chunks:
            seen_chunks.add(chunk_uid)
            merged.append(doc)

    # Sau đó mới chèn bổ sung các tài liệu bắc cầu quan hệ đồ thị vào các vị trí trống còn lại
    for doc in extra_docs:
        chunk_uid = f"{doc.metadata.get('doc_id')}_{doc.metadata.get('chunk_index', 0)}"
        if chunk_uid not in seen_chunks:
            seen_chunks.add(chunk_uid)
            merged.append(doc)
            
    return merged[:k]