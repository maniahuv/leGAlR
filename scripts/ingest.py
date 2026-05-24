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

    print("🚀 Loading content dataset...")
    content_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="content",
        split="data",
    )

    print("🚀 Loading metadata dataset...")
    metadata_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="metadata",
        split="data",
    )

    print("🚀 Loading relationships dataset for graph expansion...")
    relationship_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="relationships",
        split="data",
    )

    print("Content columns:", content_ds.column_names)
    print("Metadata columns:", metadata_ds.column_names)

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

    core_docs = []
    core_doc_ids = set()

    # Bước 1: Lọc tập văn bản chuyên ngành cốt lõi (Hôn nhân & Gia đình)
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

        core_doc_ids.add(doc_id)
        
        # Áp dụng Kỹ thuật Enrichment: Chèn Tiêu đề và Số hiệu vào nội dung thô
        title_text = meta.get("title", "")
        so_ky_hieu_text = meta.get("so_ky_hieu", "")
        enriched_content = f"Văn bản: {title_text}. Số hiệu: {so_ky_hieu_text}.\nNội dung:\n{content}"

        core_docs.append(
            Document(
                page_content=enriched_content,
                metadata={
                    "doc_id": doc_id,
                    "title": title_text,
                    "so_ky_hieu": so_ky_hieu_text,
                    "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
                    "loai_van_ban": meta.get("loai_van_ban", ""),
                    "ngay_co_hieu_luc": meta.get("ngay_co_hieu_luc", ""),
                    "ngay_het_hieu_luc": meta.get("ngay_het_hieu_luc", ""),
                    "co_quan_ban_hanh": meta.get("co_quan_ban_hanh", ""),
                    "linh_vuc": meta.get("linh_vuc", ""),
                    "nganh": meta.get("nganh", ""),
                    "tinh_trang_hieu_luc": meta.get("tinh_trang_hieu_luc", ""),
                    "source": "huggingface_core",
                },
            )
        )

    print(f"Lọc được: {len(core_docs)} tài liệu chuyên ngành gốc.")

    # Bước 2: Quét đồ thị mối quan hệ để tìm các văn bản liên đới (Sửa đổi/Thay thế/Bắc cầu)
    extended_doc_ids = set()
    for row in relationship_ds:
        src = str(row.get("doc_id", ""))
        dst = str(row.get("other_doc_id", ""))
        if src in core_doc_ids and dst:
            extended_doc_ids.add(dst)
        if dst in core_doc_ids and src:
            extended_doc_ids.add(src)

    ids_to_add = extended_doc_ids - core_doc_ids
    print(f"Tìm thấy thêm: {len(ids_to_add)} văn bản có liên kết đồ thị trực tiếp (sửa đổi/thay thế/dẫn chiếu)...")

    # Bước 3: Nạp bổ sung các tài liệu mở rộng đồ thị vào tập dữ liệu chính
    extended_counter = 0
    if ids_to_add:
        for row in content_ds:
            doc_id = str(row.get("id", ""))
            if doc_id in ids_to_add:
                meta = metadata_map.get(doc_id, {})
                content = row.get("content_html", "")
                if not content:
                    continue

                title_text = meta.get("title", "")
                so_ky_hieu_text = meta.get("so_ky_hieu", "")
                enriched_content = f"Văn bản: {title_text}. Số hiệu: {so_ky_hieu_text}.\nNội dung:\n{content}"

                core_docs.append(
                    Document(
                        page_content=enriched_content,
                        metadata={
                            "doc_id": doc_id,
                            "title": title_text,
                            "so_ky_hieu": so_ky_hieu_text,
                            "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
                            "loai_van_ban": meta.get("loai_van_ban", ""),
                            "ngay_co_hieu_luc": meta.get("ngay_co_hieu_luc", ""),
                            "ngay_het_hieu_luc": meta.get("ngay_het_hieu_luc", ""),
                            "co_quan_ban_hanh": meta.get("co_quan_ban_hanh", ""),
                            "linh_vuc": meta.get("linh_vuc", ""),
                            "nganh": meta.get("nganh", ""),
                            "tinh_trang_hieu_luc": meta.get("tinh_trang_hieu_luc", ""),
                            "source": "huggingface_graph_extension",
                        },
                    )
                )
                extended_counter += 1

    print(f"Tổng số tài liệu sau khi liên kết mở rộng đồ thị: {len(core_docs)} (Bổ sung thêm {extended_counter} tài liệu liên đới).")
    return core_docs


def main():
    print("🚀 Khởi động pipeline tải dữ liệu...")
    docs = load_legal_documents()

    # Cắt lát giới hạn để đảm bảo tài nguyên tính toán (Bạn có thể tăng lên nếu RAM khỏe)
    docs = docs[:30000]

    print(f"Loaded {len(docs)} documents into pipeline")

    if not docs:
        raise ValueError("No documents loaded. Check metadata filter or dataset columns.")

    print("🧹 Cleaning & Normalizing (NFC format)...")
    cleaned_docs = clean_documents(docs)

    print(f"Cleaned {len(cleaned_docs)} documents")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(cleaned_docs, config)

    print(f"Total chunks generated: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks created. Check cleaner or chunker.")

    print("📦 Building Chroma vector index...")
    build_chroma_index(chunks)

    print("📦 Building BM25 keyword index...")
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    print("✅ PIPELINE INGEST HOÀN THÀNH XUẤT SẮC!")
    print(f"Indexed documents: {len(docs)}")
    print(f"Indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    main()