from langchain_core.documents import Document


def dense_search(
    store,
    query: str,
    k: int = 5,
    metadata_filter: dict | None = None,
) -> list[Document]:
    """
    Tìm kiếm bằng vector similarity trong Chroma.
    """
    if metadata_filter:
        return store.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
        )

    return store.similarity_search(
        query=query,
        k=k,
    )