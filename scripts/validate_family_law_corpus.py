from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config
from src.ingestion.pdf_loader import read_jsonl, resolve_repo_path


def main() -> None:
    processed_dir = resolve_repo_path(getattr(config.dataset, "processed_dir", "data/processed/family_law"))
    metadata_path = processed_dir / "metadata.jsonl"
    content_path = processed_dir / "content.jsonl"
    chunks_path = processed_dir / "chunks.jsonl"
    relationships_path = processed_dir / "relationships.jsonl"

    metadata = read_jsonl(metadata_path)
    content = read_jsonl(content_path)
    chunks = read_jsonl(chunks_path)
    relationships = read_jsonl(relationships_path)

    if not metadata:
        raise ValueError(f"No metadata found at {metadata_path}. Run: python scripts/ingest_family_law_pdfs.py")
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}. Run: python scripts/ingest_family_law_pdfs.py")

    missing_source = [m.get("doc_id") for m in metadata if not m.get("source_url")]
    missing_effective = [m.get("doc_id") for m in metadata if not m.get("tinh_trang_hieu_luc")]

    by_doc = Counter(str(c.get("doc_id", "")) for c in chunks)
    articles_by_doc: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        doc_id = str(c.get("doc_id", ""))
        article = str(c.get("article", ""))
        if doc_id and article:
            articles_by_doc[doc_id].add(article)

    required_articles = ["81", "82", "83", "84"]
    article_presence = {
        article: any(article in articles for articles in articles_by_doc.values())
        for article in required_articles
    }

    too_short = [c.get("chunk_uid") for c in chunks if len(str(c.get("content", "")).strip()) < 80]
    too_long = [c.get("chunk_uid") for c in chunks if len(str(c.get("content", ""))) > int(config.chunking.chunk_size) * 2]

    print("===== FAMILY LAW CORPUS VALIDATION =====")
    print(f"Processed dir: {processed_dir}")
    print(f"Documents in metadata: {len(metadata)}")
    print(f"Documents in content:  {len(content)}")
    print(f"Chunks:                {len(chunks)}")
    print(f"Relationships:         {len(relationships)}")
    print("\nChunks by doc:")
    for doc_id, count in by_doc.most_common():
        print(f"- {doc_id}: {count} chunks, {len(articles_by_doc.get(doc_id, set()))} articles")

    print("\nRequired custody articles:")
    for article, ok in article_presence.items():
        print(f"- Điều {article}: {'FOUND' if ok else 'MISSING'}")

    print("\nMetadata checks:")
    print(f"- Missing source_url: {len(missing_source)} {missing_source[:10]}")
    print(f"- Missing effective status: {len(missing_effective)} {missing_effective[:10]}")
    print(f"- Very short chunks: {len(too_short)}")
    print(f"- Very long chunks:  {len(too_long)}")

    report = {
        "metadata_count": len(metadata),
        "content_count": len(content),
        "chunk_count": len(chunks),
        "relationship_count": len(relationships),
        "chunks_by_doc": dict(by_doc),
        "articles_by_doc": {k: sorted(v, key=lambda x: int(x) if x.isdigit() else 10**9) for k, v in articles_by_doc.items()},
        "required_article_presence": article_presence,
        "missing_source_url": missing_source,
        "missing_effective_status": missing_effective,
        "very_short_chunks": too_short[:100],
        "very_long_chunks": too_long[:100],
    }
    output_path = processed_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved validation report: {output_path}")


if __name__ == "__main__":
    main()
