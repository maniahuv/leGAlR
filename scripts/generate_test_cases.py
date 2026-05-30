from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset

from configs.setting import config
from src.ingestion.loader import FAMILY_LAW_KEYWORDS, html_to_text, is_family_law_document, normalize_doc_id

OUTPUT_PATH = ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json"
MANIFEST_PATH = ROOT_DIR / "data" / "evaluation" / "indexed_manifest.json"


def _load_indexed_ids() -> set[str] | None:
    if not MANIFEST_PATH.exists():
        return None
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {str(x).strip() for x in data.get("indexed_doc_ids", [])}


def load_domain_docs() -> list[dict]:
    dataset_name = getattr(config.dataset, "name", "th1nhng0/vietnamese-legal-documents")
    split = getattr(config.dataset, "split", "data")
    indexed_ids = _load_indexed_ids()

    metadata_ds = load_dataset(dataset_name, name="metadata", split=split)
    content_ds = load_dataset(dataset_name, name="content", split=split)
    metadata_map = {normalize_doc_id(row.get("id")): row for row in metadata_ds}

    docs: list[dict] = []
    for row in content_ds:
        doc_id = normalize_doc_id(row.get("id"))
        if indexed_ids is not None and doc_id not in indexed_ids:
            continue
        html = row.get("content_html", "") or ""
        if not doc_id or not html:
            continue
        meta = metadata_map.get(doc_id, {})
        content = html_to_text(html)
        if indexed_ids is None and not is_family_law_document(meta, content):
            continue
        docs.append({
            "doc_id": doc_id,
            "title": meta.get("title", "") or "",
            "so_ky_hieu": meta.get("so_ky_hieu", "") or "",
            "loai_van_ban": meta.get("loai_van_ban", "") or "",
            "ngay_ban_hanh": meta.get("ngay_ban_hanh", "") or "",
            "ngay_co_hieu_luc": meta.get("ngay_co_hieu_luc", "") or "",
            "ngay_het_hieu_luc": meta.get("ngay_het_hieu_luc", "") or "",
            "tinh_trang_hieu_luc": meta.get("tinh_trang_hieu_luc", "") or "",
            "content": content,
        })
    print(f"Loaded domain docs for test generation: {len(docs)}")
    return docs


def find_docs(docs, must_have=None, title_contains=None, so_ky_hieu=None, only_effective=False):
    results = []
    for doc in docs:
        text = " ".join([doc["title"], doc["so_ky_hieu"], doc["content"][:8000]]).lower()
        if must_have and not all(kw.lower() in text for kw in must_have):
            continue
        if title_contains and title_contains.lower() not in doc["title"].lower():
            continue
        if so_ky_hieu and so_ky_hieu.lower() not in doc["so_ky_hieu"].lower():
            continue
        if only_effective and doc.get("tinh_trang_hieu_luc") != "Còn hiệu lực":
            continue
        results.append(doc)
    return results


def make_case(case_id, scenario, query, relevant_docs, reference_answer):
    relevant_ids = []
    for doc in relevant_docs:
        did = str(doc.get("doc_id", "")).strip()
        if did and did not in relevant_ids:
            relevant_ids.append(did)
    if not relevant_ids:
        return None
    return {
        "id": f"tc_{case_id:03d}",
        "scenario": scenario,
        "query": query,
        "relevant_ids": relevant_ids,
        "reference_answer": reference_answer,
    }


def add_case(cases, cid, scenario, query, docs, answer):
    case = make_case(cid, scenario, query, docs, answer)
    if case:
        cases.append(case)
        return cid + 1
    return cid


def main():
    docs = load_domain_docs()
    if not docs:
        raise ValueError("Không có docs để sinh test. Hãy chạy python scripts/ingest.py trước.")

    cases = []
    cid = 1

    law_2014 = find_docs(docs, title_contains="Hôn nhân và gia đình", so_ky_hieu="52/2014/QH13")[:1]
    law_2000 = find_docs(docs, title_contains="Hôn nhân và gia đình", so_ky_hieu="22/2000/QH10")[:1]
    guide_2016 = find_docs(docs, so_ky_hieu="01/2016/TTLT-TANDTC-VKSNDTC-BTP")[:1]
    nd_126 = find_docs(docs, so_ky_hieu="126/2014/NĐ-CP")[:1]
    nd_82 = find_docs(docs, so_ky_hieu="82/2020/NĐ-CP")[:1]
    ho_tich_2014 = find_docs(docs, title_contains="Hộ tịch", so_ky_hieu="60/2014/QH13")[:1]

    semantic_questions = [
        ("Độ tuổi hợp pháp để nam nữ kết hôn là bao nhiêu?", "Nam từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên."),
        ("Điều kiện kết hôn theo Luật Hôn nhân và gia đình 2014 là gì?", "Điều kiện kết hôn gồm: đủ tuổi, tự nguyện, không mất năng lực hành vi dân sự và không thuộc trường hợp cấm kết hôn."),
        ("Kết hôn trái pháp luật là gì?", "Kết hôn trái pháp luật là việc đăng ký kết hôn vi phạm điều kiện kết hôn."),
        ("Tảo hôn là gì?", "Tảo hôn là việc lấy vợ, lấy chồng khi một bên hoặc cả hai bên chưa đủ tuổi kết hôn."),
        ("Nhà nước có thừa nhận hôn nhân giữa những người cùng giới tính không?", "Nhà nước không thừa nhận hôn nhân giữa những người cùng giới tính."),
        ("Chồng có quyền yêu cầu ly hôn khi vợ đang mang thai không?", "Chồng không có quyền yêu cầu ly hôn khi vợ đang có thai, sinh con hoặc nuôi con dưới 12 tháng tuổi."),
        ("Con dưới 36 tháng tuổi khi ly hôn thường được giao cho ai nuôi?", "Con dưới 36 tháng tuổi thường được giao cho mẹ trực tiếp nuôi, trừ trường hợp luật định hoặc có thỏa thuận khác."),
        ("Con từ bao nhiêu tuổi thì Tòa án phải xem xét nguyện vọng khi cha mẹ ly hôn?", "Con từ đủ 07 tuổi trở lên thì phải xem xét nguyện vọng của con."),
        ("Tài sản chung của vợ chồng gồm những tài sản nào?", "Tài sản chung gồm tài sản do vợ chồng tạo ra và các thu nhập, hoa lợi, lợi tức trong thời kỳ hôn nhân theo luật."),
        ("Mức cấp dưỡng do ai quyết định?", "Mức cấp dưỡng do các bên thỏa thuận; nếu không thỏa thuận được thì yêu cầu Tòa án giải quyết."),
    ]
    for q, a in semantic_questions:
        cid = add_case(cases, cid, "single-hop-semantic", q, law_2014, a)

    article_numbers = [8, 10, 33, 43, 51, 56, 59, 81, 82, 83, 95, 107, 110, 116, 117]
    for n in article_numbers:
        cid = add_case(cases, cid, "exact-keyword-fact", f"Điều {n} Luật Hôn nhân và gia đình 2014 quy định gì?", law_2014, f"Điều {n} quy định nội dung tương ứng trong Luật Hôn nhân và gia đình 2014.")

    for q, docs_ref, a in [
        ("Thông tư liên tịch 01/2016 hướng dẫn nội dung nào của Luật Hôn nhân và gia đình?", guide_2016, "Thông tư liên tịch 01/2016 hướng dẫn một số vấn đề về hôn nhân và gia đình."),
        ("Nghị định 126/2014/NĐ-CP quy định chi tiết thi hành luật nào?", nd_126, "Nghị định 126/2014/NĐ-CP quy định chi tiết một số điều của Luật Hôn nhân và gia đình."),
        ("Nghị định 82/2020/NĐ-CP có quy định xử phạt hành chính về hôn nhân gia đình không?", nd_82, "Nghị định 82/2020/NĐ-CP có nội dung liên quan xử phạt vi phạm hành chính trong lĩnh vực hôn nhân và gia đình."),
        ("Luật Hộ tịch 2014 có liên quan đến đăng ký kết hôn không?", ho_tich_2014, "Luật Hộ tịch 2014 có quy định về đăng ký kết hôn và các việc hộ tịch."),
    ]:
        cid = add_case(cases, cid, "exact-keyword-fact", q, docs_ref, a)

    if law_2014 and law_2000:
        cid = add_case(cases, cid, "multi-hop-graph", "Luật Hôn nhân và Gia đình năm 2014 thay thế Luật Hôn nhân và Gia đình năm nào?", [law_2014[0], law_2000[0]], "Luật Hôn nhân và Gia đình năm 2014 thay thế Luật Hôn nhân và Gia đình năm 2000.")
        cid = add_case(cases, cid, "multi-hop-graph", "Khi trả lời về điều kiện kết hôn hiện hành nên ưu tiên Luật HNGĐ 2014 hay Luật HNGĐ 2000?", [law_2014[0], law_2000[0]], "Nên ưu tiên Luật Hôn nhân và Gia đình 2014 nếu văn bản này đang là văn bản hiện hành.")

    for doc in docs:
        if len(cases) >= 100:
            break
        title = doc.get("title", "")
        if not title:
            continue
        # Chỉ tạo title query cho văn bản có từ khóa miền rõ ràng để tránh gold nhiễu.
        text = (title + " " + doc.get("content", "")[:1500]).lower()
        if not any(kw in text for kw in FAMILY_LAW_KEYWORDS):
            continue
        q = f"Văn bản {title} số {doc.get('so_ky_hieu', '')} quy định nội dung gì liên quan đến hôn nhân gia đình?"
        a = f"Văn bản {title} số {doc.get('so_ky_hieu', '')} có nội dung liên quan đến hôn nhân và gia đình."
        cid = add_case(cases, cid, "metadata-title-query", q, [doc], a)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} test cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
