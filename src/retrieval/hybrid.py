from langchain_core.documents import Document
from src.retrieval.dense import dense_search


def _doc_key(doc: Document) -> str:
    """
    Tạo key để chống trùng document.
    """
    metadata = doc.metadata or {}

    return f"{metadata.get('doc_id', '')}_{metadata.get('chunk_index', 0)}"


def hybrid_search(
    store,
    bm25,
    query: str,
    k: int = 5,
    dense_k: int | None = None,
    bm25_k: int | None = None,
) -> list[Document]:
    """
    Hybrid search = dense search + BM25 search.

    Dense search mạnh về ngữ nghĩa.
    BM25 mạnh về keyword, số điều, tên văn bản, năm ban hành.
    """
    dense_k = dense_k or k
    bm25_k = bm25_k or k

    dense_docs = dense_search(store, query, k=dense_k)

    bm25.k = bm25_k
    bm25_docs = bm25.invoke(query)

    merged: list[Document] = []
    seen = set()

    # Ưu tiên xen kẽ dense và bm25
    for docs in zip(dense_docs, bm25_docs):
        for doc in docs:
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    # Nếu một bên dài hơn bên kia
    for doc in dense_docs + bm25_docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged[:k]