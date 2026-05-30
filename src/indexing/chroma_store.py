from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from configs.setting import config
from src.indexing.embeddings import get_embedding_model


def get_store() -> Chroma:
    Path(config.vector_store.persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.vector_store.collection_name,
        persist_directory=config.vector_store.persist_directory,
        embedding_function=get_embedding_model(),
    )


def _ids_for_batch(batch: list[Document]) -> list[str]:
    ids = []
    seen = set()
    for i, doc in enumerate(batch):
        meta = doc.metadata or {}
        base = str(meta.get("chunk_uid") or f"{meta.get('doc_id', 'doc')}_{meta.get('chunk_index', i)}")
        cid = base
        n = 1
        while cid in seen:
            n += 1
            cid = f"{base}_{n}"
        seen.add(cid)
        ids.append(cid)
    return ids


def build_chroma_index(docs: list[Document], batch_size: int = 256, reset: bool | None = None) -> Chroma:
    if reset is None:
        reset = bool(getattr(config.vector_store, "reset_on_ingest", True))

    store = get_store()
    if reset:
        try:
            store.delete_collection()
        except Exception:
            pass
        store = get_store()

    if not docs:
        print("No documents to add to Chroma.")
        return store

    print(f"Adding {len(docs)} chunks to Chroma...")
    for i in tqdm(range(0, len(docs), batch_size), desc="Chroma indexing"):
        batch = docs[i:i + batch_size]
        store.add_documents(batch, ids=_ids_for_batch(batch))
    return store
