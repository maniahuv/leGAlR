from pathlib import Path

from langchain_core.tools import tool
from langchain_core.documents import Document

from configs.setting import config

from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents

from src.indexing.chroma_store import build_chroma_index
from src.indexing.bm25_index import build_bm25_index, save_bm25_index


def _load_documents_from_path(path: str) -> list[Document]:
    """
    Load documents from a local path.

    Hiện tại hàm này giả định loader.py của bạn có một trong các hàm:
    - load_documents
    - load_pdf_documents
    - load_text_documents

    Nếu loader.py chưa có, cần bổ sung sau.
    """
    try:
        from src.ingestion.loader import load_documents

        return load_documents(path)

    except ImportError as e:
        raise ImportError(
            "Không tìm thấy hàm load_documents trong src.ingestion.loader. "
            "Bạn cần định nghĩa hàm load_documents(path) trong loader.py "
            "hoặc chỉnh ingestion_tools.py theo đúng tên hàm loader bạn đang có."
        ) from e


@tool
def ingestion_tool(path: str) -> dict:
    """
    Load legal documents from a local path, clean them, split them into chunks,
    then build both Chroma vector index and BM25 keyword index.

    Use this tool when the user wants to ingest new legal documents
    or rebuild the retrieval indexes from local source files.
    """
    try:
        input_path = Path(path)

        if not input_path.exists():
            return {
                "status": "error",
                "message": f"Path does not exist: {path}",
            }

        docs = _load_documents_from_path(str(input_path))

        if not docs:
            return {
                "status": "error",
                "message": "No documents loaded from the given path.",
                "path": str(input_path),
            }

        cleaned_docs = clean_documents(docs)

        if not cleaned_docs:
            return {
                "status": "error",
                "message": "No documents left after cleaning.",
                "input_documents": len(docs),
            }

        chunks = chunk_documents(cleaned_docs, config)

        if not chunks:
            return {
                "status": "error",
                "message": "No chunks created. Check chunker config.",
                "input_documents": len(docs),
                "cleaned_documents": len(cleaned_docs),
            }

        build_chroma_index(chunks)

        bm25 = build_bm25_index(chunks)
        save_bm25_index(bm25)

        return {
            "status": "success",
            "path": str(input_path),
            "input_documents": len(docs),
            "cleaned_documents": len(cleaned_docs),
            "chunks": len(chunks),
            "indexes": ["chroma", "bm25"],
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# Alias nếu code cũ còn import tên này
ingest_documents_tool = ingestion_tool