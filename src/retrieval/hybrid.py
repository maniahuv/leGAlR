from langchain_core.documents import Document
from src.retrieval.dense import dense_search

def hybrid_search(
    store,
    bm25,
    query: str,
    k: int = 5,
    dense_k: int | None = None,
    bm25_k: int | None = None,
) -> list[Document]:
    # Thu hồi diện rộng ở tầng chunk (k * 10) để chống hiện tượng "Cạnh tranh Chunk"
    POOL_K = k * 10
    
    dense_docs = dense_search(store, query, k=POOL_K)
    
    bm25.k = POOL_K
    bm25_docs = bm25.invoke(query)

    merged: list[Document] = []
    seen_văn_bản = set()

    # Ưu tiên gom các tài liệu từ BM25 trước (đặc biệt hiệu quả cho nhóm tên văn bản cổ/sắc lệnh)
    for doc in bm25_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", ""))
        if doc_id and doc_id not in seen_văn_bản:
            seen_văn_bản.add(doc_id)
            merged.append(doc)

    # Bổ sung các tài liệu từ Dense Search nếu chưa xuất hiện
    for doc in dense_docs:
        doc_id = str((doc.metadata or {}).get("doc_id", ""))
        if doc_id and doc_id not in seen_văn_bản:
            seen_văn_bản.add(doc_id)
            merged.append(doc)

    return merged[:k]