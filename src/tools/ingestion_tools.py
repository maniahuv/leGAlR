from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from configs.setting import config
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.loader import load_documents
from src.indexing.bm25_index import build_bm25_index, save_bm25_index
from src.indexing.chroma_store import build_chroma_index


@tool
def ingestion_tool(path: str) -> dict:
    """Load local legal documents, clean, chunk, and rebuild Chroma + BM25 indexes."""
    try:
        input_path = Path(path)
        if not input_path.exists():
            return {"status": "error", "message": f"Path does not exist: {path}"}
        docs = load_documents(str(input_path))
        if not docs:
            return {"status": "error", "message": "No supported documents found."}
        cleaned_docs = clean_documents(docs)
        chunks = chunk_documents(cleaned_docs, config)
        if not chunks:
            return {"status": "error", "message": "No chunks created."}
        build_chroma_index(chunks, reset=getattr(config.vector_store, "reset_on_ingest", True))
        bm25 = build_bm25_index(chunks)
        save_bm25_index(bm25)
        return {"status": "success", "input_documents": len(docs), "chunks": len(chunks), "indexes": ["chroma", "bm25"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


ingest_documents_tool = ingestion_tool
