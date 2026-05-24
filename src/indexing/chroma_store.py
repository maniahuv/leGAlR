from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.indexing.embeddings import get_embedding_model
from configs.setting import config

def get_store() -> Chroma:
    """
    Load hoặc tạo Chroma vector store.
    """

    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=config.vector_store.collection_name,
        persist_directory=config.vector_store.persist_directory,
        embedding_function=embedding_model,
    )

# def build_chroma_index(docs: list[Document]) -> Chroma:
#     """
#     Build vector index từ danh sách Document.
#     """

#     store = get_store()

#     if docs:
#         store.add_documents(docs)

#     return store


from tqdm import tqdm


def build_chroma_index(docs, batch_size: int = 256):
    store = get_store()

    if not docs:
        print("No documents to add to Chroma.")
        return store

    print(f"Adding {len(docs)} chunks to Chroma...")

    for i in tqdm(range(0, len(docs), batch_size), desc="Chroma indexing"):
        batch = docs[i:i + batch_size]
        store.add_documents(batch)

    return store