import networkx as nx
from langchain_core.documents import Document
from src.retrieval.dense import dense_search



def build_graph(relationships: list[dict]) -> nx.DiGraph: #dict là dictionary dạng key-value giống json
    graph=nx.DiGraph() # khởi tạo graph 
    for row in relationships:
        src=str(row.get("doc_id", ""))
        dst=str(row.get("other_doc_id", ""))
        rel_type=row.get("relationship", "")
        if src and dst: #nếu có vector giữa 2 node thì thêm vào graph quan hệ của cạnh
            graph.add_edge(src, dst, rel_type=rel_type)
    return graph

def graph_search(store, graph: nx.DiGraph, query: str, k: int=5, initial_k:int=3, max_hops:int=2):
    seed_docs=dense_search(store, query, k=initial_k) # tìm các docs đóng vai trò làm hạt nhân tìm kiếm bằng vector search 
    seed_ids={d.metadata.get("doc_id", "") 
              for d in seed_docs
              if d.metadata.get("doc_id")} # lấy các ids dựa trên các docs đã tìm thấy ở trên, cấu trúc này là set comprehension -> trả về 1 set, nếu để () thì nó sẽ trả lần lượt mỗi phần tử 

    reachable, frontier = set(seed_ids), set(seed_ids) #reachable: các node đã biết, frontier: các node dùng để xét vòng tiếp theo
    for _ in range(max_hops): 
        nxt: set[str] =set() # tạo 1 biến nxt, kiểu set[str], ban đầu rỗng
        for node in frontier: 
            if node not in graph: # lần đầu có thể vector search ra các seed_ids sẽ có thể ra các node không nằm trong graph vì graph chỉ chứa các docs có relationship 
                continue
            nxt |={nb # union nxt |= nb = nxt hợp nb 
                   for _, nb, _ in graph.out_edges(node, data=True) # cú pháp unpack tuple, chỉ lấy phần tử ở giữa
                   if nb not in reachable} #set comprehension, thêm các node mới được truy xuất tới từ node hiện tại (tất nhiên là không nằm trong tập reachable đã biết)
            nxt |={nb # union nxt |= = nxt hợp nb 
                   for nb, _, _ in graph.in_edges(node, data=True) # cú pháp unpack tuple, chỉ lấy phần tử ở giữa 
                   if nb not in reachable} #set comprehension, thêm các node mới được truy xuất tới node hiện tại 
        reachable |= nxt # reachable = reachable hợp nxt (next)
        frontier = nxt #nxt chứa các node không nằm trong reachable tức các node dùng để tìm các node mới 
        if not frontier: #không còn node mới thì phá vòng lặp 
            break
    
    extra_ids = reachable - seed_ids # mục đích để lọc ra các docs mới không phải docs_seed ban đầu, lọc theo id 
    extra_docs = dense_search(store, query, k=k*2,
                              metadata_filter={"doc_id": {"$in": list(extra_ids)}}) if extra_ids else []
    
    # loại bỏ trùng lặp 
    seen, deduped = set(), [] #seen kiểm tra đã gặp chưa, deduped lưu kết quả 
    for doc in list(seed_docs) + extra_docs:
        key=f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index',0)}"
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped[:k] #lấy k phần tử đầu, không tính phần tử có index = k

