from langchain_core.documents import Document
from datasets import load_dataset

def load_document(config, sample_size=None):
    content_rows = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="content",
        split="data"
    )
    metadata_rows = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="metadata",
        split="data"
    )
    meta_lookup={
        str(meta["id"]): meta # tạo key:value với mỗi meta trong metadata 
        for meta in metadata_rows
    }
    if sample_size:
        content_rows=content_rows.select(range(sample_size))
    docs=[]
    for row in content_rows:
        doc_id = str(row["id"]) 
        meta = meta_lookup.get(doc_id, {}) #meta_lookup thường dùng key dạng chuỗi
        content = row.get("content_html","")
        if not content.strip():
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    # 📄 Thông tin cơ bản
                    "official_number": meta.get("so_ky_hieu", ""),
                    "issue_date": meta.get("ngay_ban_hanh", ""),
                    "doc_type": meta.get("loai_van_ban", ""),

                    # ⏱️ Hiệu lực
                    "effective_date": meta.get("ngay_co_hieu_luc", ""),
                    "expiry_date": meta.get("ngay_het_hieu_luc", ""),
                    "status": meta.get("tinh_trang_hieu_luc", ""),

                    # 🏛️ Nguồn & công bố
                    "source": meta.get("nguon_thu_thap", ""),
                    "gazette_date": meta.get("ngay_dang_cong_bao", ""),

                    # 🧠 Phân loại
                    "sector": meta.get("nganh", ""),
                    "legal_field": meta.get("linh_vuc", ""),

                    # 🏢 Cơ quan & người ký
                    "authority": meta.get("co_quan_ban_hanh", ""),
                    "signatory_title": meta.get("chuc_danh", ""),
                    "signatory_name": meta.get("nguoi_ky", ""),

                    # 🌍 Phạm vi
                    "scope": meta.get("pham_vi", ""),

                    # 📌 Ghi chú
                    "implementation_note": meta.get("thong_tin_ap_dung", ""),                    
                    
                    "article": meta.get("dieu",""),
                    "chapter": meta.get("chuong",""),
                },
            )
        )
    return docs
