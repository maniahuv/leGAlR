import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config

from langchain_core.documents import Document
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.indexing.chroma_store import build_chroma_index
from src.indexing.bm25_index import build_bm25_index, save_bm25_index


def load_legal_documents() -> list[Document]:
    from datasets import load_dataset

    print("Loading content dataset...")
    content_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="content",
        split="data",
    )

    print("Loading metadata dataset...")
    metadata_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="metadata",
        split="data",
    )

    print("Content columns:", content_ds.column_names)
    print("Metadata columns:", metadata_ds.column_names)
    print("First content id:", content_ds[0]["id"])

    metadata_map = {
        str(row["id"]): row
        for row in metadata_ds
    }

    target_keywords = [
        "hôn nhân",
        "gia đình",
        "ly hôn",
        "kết hôn",
        "vợ chồng",
        "nuôi con",
        "cấp dưỡng",
        "con chung",
        "tài sản chung",
        "tài sản riêng",
    ]

    docs = []

    for row in content_ds:
        doc_id = str(row.get("id", ""))
        content = row.get("content_html", "")

        if not doc_id or not content:
            continue

        meta = metadata_map.get(doc_id, {})

        filter_text = " ".join([
            str(meta.get("title", "")),
            str(meta.get("linh_vuc", "")),
            str(meta.get("nganh", "")),
        ]).lower()

        if not any(keyword in filter_text for keyword in target_keywords):
            continue

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "so_ky_hieu": meta.get("so_ky_hieu", ""),
                    "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
                    "loai_van_ban": meta.get("loai_van_ban", ""),
                    "ngay_co_hieu_luc": meta.get("ngay_co_hieu_luc", ""),
                    "ngay_het_hieu_luc": meta.get("ngay_het_hieu_luc", ""),
                    "co_quan_ban_hanh": meta.get("co_quan_ban_hanh", ""),
                    "linh_vuc": meta.get("linh_vuc", ""),
                    "nganh": meta.get("nganh", ""),
                    "tinh_trang_hieu_luc": meta.get("tinh_trang_hieu_luc", ""),
                    "source": "huggingface",
                },
            )
        )

    return docs


def main():
    print("🚀 Loading dataset...")
    docs = load_legal_documents()

    # Test trước cho nhẹ. Khi ổn thì bỏ dòng này.
    docs = docs[:30000]

    print(f"Loaded {len(docs)} documents")

    if not docs:
        raise ValueError("No documents loaded. Check metadata filter or dataset columns.")

    print("🧹 Cleaning...")
    cleaned_docs = clean_documents(docs)

    print(f"Cleaned {len(cleaned_docs)} documents")

    print("✂️ Chunking...")
    chunks = chunk_documents(cleaned_docs, config)

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created. Check cleaner or chunker.")

    print("📦 Building Chroma index...")
    build_chroma_index(chunks)

    print("📦 Building BM25 index...")
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    print("✅ DONE!")
    print(f"Indexed documents: {len(docs)}")
    print(f"Indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    main()