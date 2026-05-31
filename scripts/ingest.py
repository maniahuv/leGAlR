from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config
from src.ingestion.loader import load_hf_legal_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.indexing.chroma_store import build_chroma_index
from src.indexing.bm25_index import build_bm25_index, save_bm25_index


def write_index_manifest(chunks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc_ids = sorted({str((c.metadata or {}).get("doc_id", "")).strip() for c in chunks if (c.metadata or {}).get("doc_id")})
    payload = {
        "indexed_doc_count": len(doc_ids),
        "indexed_chunk_count": len(chunks),
        "indexed_doc_ids": doc_ids,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    print("🚀 Loading Vietnamese legal documents from HuggingFace...")
    docs = load_hf_legal_documents(config, sample_size=getattr(config.dataset, "sample_size", None))
    if not docs:
        raise ValueError("No documents loaded. Check dataset availability and domain filter.")

    print("🧹 Cleaning HTML/text while preserving legal structure...")
    cleaned_docs = clean_documents(docs)
    if not cleaned_docs:
        raise ValueError("No documents left after cleaning.")

    print("✂️ Chunking by legal structure...")
    chunks = chunk_documents(cleaned_docs, config)
    if not chunks:
        raise ValueError("No chunks created. Check chunker config.")

    print(f"Loaded documents: {len(docs)}")
    print(f"Generated chunks: {len(chunks)}")

    print("📦 Building Chroma vector index...")
    build_chroma_index(chunks, reset=getattr(config.vector_store, "reset_on_ingest", True))

    print("📦 Building Vietnamese BM25 index...")
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    write_index_manifest(chunks, ROOT_DIR / "data" / "evaluation" / "indexed_manifest.json")

    print("✅ INGEST DONE")
    print("Next: python scripts/generate_test_cases.py")
    print("Then: python scripts/run_benchmark.py")


if __name__ == "__main__":
    main()
