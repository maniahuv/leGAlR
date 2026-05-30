from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from datasets import load_dataset
from langchain_core.documents import Document

FAMILY_LAW_KEYWORDS = [
    "hôn nhân", "gia đình", "ly hôn", "kết hôn", "vợ chồng", "chồng", "vợ",
    "nuôi con", "quyền nuôi con", "cấp dưỡng", "con chung", "con riêng",
    "tài sản chung", "tài sản riêng", "chế độ tài sản", "chia tài sản",
    "tảo hôn", "kết hôn giả tạo", "cưỡng ép kết hôn", "cấm kết hôn",
    "mang thai hộ", "sinh con", "cha mẹ", "xác định cha mẹ con",
    "nhận cha", "nhận mẹ", "nhận con", "giám hộ", "nuôi con nuôi",
    "hộ tịch", "đăng ký kết hôn", "khai sinh", "bạo lực gia đình", "trẻ em",
]


def normalize_doc_id(value) -> str:
    return str(value or "").strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _meta_value(meta: dict, key: str) -> str:
    value = meta.get(key, "") if meta else ""
    return "" if value is None else str(value)


def build_doc_metadata(doc_id: str, meta: dict, source: str = "huggingface") -> dict:
    return {
        "doc_id": doc_id,
        "title": _meta_value(meta, "title"),
        "so_ky_hieu": _meta_value(meta, "so_ky_hieu"),
        "loai_van_ban": _meta_value(meta, "loai_van_ban"),
        "ngay_ban_hanh": _meta_value(meta, "ngay_ban_hanh"),
        "ngay_co_hieu_luc": _meta_value(meta, "ngay_co_hieu_luc"),
        "ngay_het_hieu_luc": _meta_value(meta, "ngay_het_hieu_luc"),
        "tinh_trang_hieu_luc": _meta_value(meta, "tinh_trang_hieu_luc"),
        "nguon_thu_thap": _meta_value(meta, "nguon_thu_thap"),
        "ngay_dang_cong_bao": _meta_value(meta, "ngay_dang_cong_bao"),
        "nganh": _meta_value(meta, "nganh"),
        "linh_vuc": _meta_value(meta, "linh_vuc"),
        "co_quan_ban_hanh": _meta_value(meta, "co_quan_ban_hanh"),
        "chuc_danh": _meta_value(meta, "chuc_danh"),
        "nguoi_ky": _meta_value(meta, "nguoi_ky"),
        "pham_vi": _meta_value(meta, "pham_vi"),
        "thong_tin_ap_dung": _meta_value(meta, "thong_tin_ap_dung"),
        "source": source,
    }


def is_family_law_document(meta: dict, content_text: str, content_filter_chars: int = 6000) -> bool:
    haystack = " ".join([
        _meta_value(meta, "title"),
        _meta_value(meta, "so_ky_hieu"),
        _meta_value(meta, "loai_van_ban"),
        _meta_value(meta, "linh_vuc"),
        _meta_value(meta, "nganh"),
        content_text[:content_filter_chars],
    ]).lower()
    return any(keyword in haystack for keyword in FAMILY_LAW_KEYWORDS)


def _relationship_neighbors(relationships: Iterable[dict], seed_ids: set[str], hops: int, limit: int) -> set[str]:
    if not seed_ids or hops <= 0 or limit <= 0:
        return set()

    adjacency: dict[str, set[str]] = {}
    for row in relationships:
        src = normalize_doc_id(row.get("doc_id"))
        dst = normalize_doc_id(row.get("other_doc_id"))
        if not src or not dst:
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    visited = set(seed_ids)
    frontier = set(seed_ids)
    extra: set[str] = set()

    for _ in range(hops):
        nxt: set[str] = set()
        for node in frontier:
            for nb in adjacency.get(node, set()):
                if nb not in visited:
                    visited.add(nb)
                    extra.add(nb)
                    nxt.add(nb)
                    if len(extra) >= limit:
                        return extra
        frontier = nxt
        if not frontier:
            break
    return extra


def load_hf_legal_documents(config, sample_size: int | None = None) -> list[Document]:
    """Load metadata + content + optional relationship expansion từ HuggingFace dataset."""
    dataset_name = getattr(config.dataset, "name", "th1nhng0/vietnamese-legal-documents")
    split = getattr(config.dataset, "split", "data")

    print("Loading metadata dataset...")
    metadata_ds = load_dataset(dataset_name, name=getattr(config.dataset, "metadata_config", "metadata"), split=split)
    print("Loading content dataset...")
    content_ds = load_dataset(dataset_name, name=getattr(config.dataset, "content_config", "content"), split=split)

    if sample_size:
        content_ds = content_ds.select(range(min(sample_size, len(content_ds))))

    metadata_map = {normalize_doc_id(row.get("id")): row for row in metadata_ds}

    selected_rows: dict[str, tuple[dict, str, str]] = {}
    seed_ids: set[str] = set()

    for row in content_ds:
        doc_id = normalize_doc_id(row.get("id"))
        content_html = row.get("content_html", "") or ""
        if not doc_id or not content_html:
            continue
        meta = metadata_map.get(doc_id, {})
        content_text = html_to_text(content_html)
        if is_family_law_document(meta, content_text, getattr(config.dataset, "content_filter_chars", 6000)):
            selected_rows[doc_id] = (meta, content_html, "huggingface_core")
            seed_ids.add(doc_id)

    print(f"Core family-law documents: {len(seed_ids)}")

    if getattr(config.dataset, "include_graph_neighbors", True) and seed_ids:
        print("Loading relationship dataset for graph expansion...")
        relationships_ds = load_dataset(dataset_name, name=getattr(config.dataset, "relationships_config", "relationships"), split=split)
        extra_ids = _relationship_neighbors(
            relationships_ds,
            seed_ids=seed_ids,
            hops=int(getattr(config.dataset, "graph_expansion_hops", 1)),
            limit=int(getattr(config.dataset, "graph_expansion_max_docs", 800)),
        )
        missing_extra = extra_ids - set(selected_rows)
        print(f"Graph-neighbor documents found: {len(missing_extra)}")
        if missing_extra:
            missing_set = set(missing_extra)
            for row in content_ds:
                doc_id = normalize_doc_id(row.get("id"))
                if doc_id in missing_set and row.get("content_html"):
                    selected_rows[doc_id] = (metadata_map.get(doc_id, {}), row.get("content_html", ""), "huggingface_graph_neighbor")

    docs: list[Document] = []
    for doc_id, (meta, content_html, source) in selected_rows.items():
        metadata = build_doc_metadata(doc_id, meta, source=source)
        header = (
            f"Văn bản: {metadata['title']}\n"
            f"Số hiệu: {metadata['so_ky_hieu']}\n"
            f"Loại văn bản: {metadata['loai_van_ban']}\n"
            f"Cơ quan ban hành: {metadata['co_quan_ban_hanh']}\n"
            f"Ngày ban hành: {metadata['ngay_ban_hanh']}\n"
            f"Ngày có hiệu lực: {metadata['ngay_co_hieu_luc']}\n"
            f"Ngày hết hiệu lực: {metadata['ngay_het_hieu_luc']}\n"
            f"Tình trạng hiệu lực: {metadata['tinh_trang_hieu_luc']}\n"
            "Nội dung:\n"
        )
        docs.append(Document(page_content=header + content_html, metadata=metadata))

    max_docs = getattr(config.dataset, "max_docs", None)
    if max_docs:
        docs = docs[: int(max_docs)]
    print(f"Total documents loaded for ingest: {len(docs)}")
    return docs


def load_documents(path: str) -> list[Document]:
    """Load local .pdf/.txt/.md files. Dùng cho ingestion_tool."""
    input_path = Path(path)
    files: list[Path]
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}])

    docs: list[Document] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            for doc in PyPDFLoader(str(file)).load():
                doc.metadata.setdefault("source", str(file))
                doc.metadata.setdefault("doc_id", file.stem)
                docs.append(doc)
        else:
            text = file.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": str(file), "doc_id": file.stem, "title": file.name}))
    return docs


def load_document(config, sample_size=None):
    """Backward-compatible alias cũ."""
    return load_hf_legal_documents(config, sample_size=sample_size)
