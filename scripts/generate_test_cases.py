import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset
from bs4 import BeautifulSoup


OUTPUT_PATH = ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json"


TARGET_KEYWORDS = [
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
    "tảo hôn",
    "mang thai hộ",
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_domain_docs():
    print("Loading metadata...")
    metadata_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="metadata",
        split="data",
    )

    print("Loading content...")
    content_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        name="content",
        split="data",
    )

    metadata_map = {
        str(row["id"]): row
        for row in metadata_ds
    }

    docs = []

    for row in content_ds:
        doc_id = str(row.get("id", ""))
        content_html = row.get("content_html", "")

        if not doc_id or not content_html:
            continue

        meta = metadata_map.get(doc_id, {})
        text_for_filter = " ".join([
            str(meta.get("title", "")),
            str(meta.get("linh_vuc", "")),
            str(meta.get("nganh", "")),
            str(meta.get("so_ky_hieu", "")),
        ]).lower()

        content_text = clean_html(content_html)

        combined = (text_for_filter + " " + content_text[:2000].lower())

        if not any(keyword in combined for keyword in TARGET_KEYWORDS):
            continue

        docs.append({
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "so_ky_hieu": meta.get("so_ky_hieu", ""),
            "loai_van_ban": meta.get("loai_van_ban", ""),
            "ngay_ban_hanh": meta.get("ngay_ban_hanh", ""),
            "ngay_co_hieu_luc": meta.get("ngay_co_hieu_luc", ""),
            "tinh_trang_hieu_luc": meta.get("tinh_trang_hieu_luc", ""),
            "content": content_text,
        })

    print(f"Loaded {len(docs)} domain docs")
    return docs


def find_docs(docs, must_have=None, title_contains=None, so_ky_hieu=None, only_effective=False):
    results = []

    for doc in docs:
        text = " ".join([
            doc["title"],
            doc["so_ky_hieu"],
            doc["content"][:5000],
        ]).lower()

        if must_have and not all(keyword.lower() in text for keyword in must_have):
            continue

        if title_contains and title_contains.lower() not in doc["title"].lower():
            continue

        if so_ky_hieu and so_ky_hieu.lower() not in doc["so_ky_hieu"].lower():
            continue

        if only_effective and doc["tinh_trang_hieu_luc"] != "Còn hiệu lực":
            continue

        results.append(doc)

    return results


def make_case(case_id, scenario, query, relevant_docs, reference_answer):
    relevant_ids = []
    for doc in relevant_docs:
        doc_id = str(doc["doc_id"])
        if doc_id not in relevant_ids:
            relevant_ids.append(doc_id)

    return {
        "id": f"tc_{case_id:03d}",
        "scenario": scenario,
        "query": query,
        "relevant_ids": relevant_ids,
        "reference_answer": reference_answer,
    }


def main():
    docs = load_domain_docs()

    cases = []
    cid = 1

    law_2014 = find_docs(
        docs,
        title_contains="Hôn nhân và gia đình",
        so_ky_hieu="52/2014/QH13",
    )

    law_2000 = find_docs(
        docs,
        title_contains="Hôn nhân và gia đình",
        so_ky_hieu="22/2000/QH10",
    )

    guide_2016 = find_docs(
        docs,
        so_ky_hieu="01/2016/TTLT-TANDTC-VKSNDTC-BTP",
    )

    nd_126 = find_docs(
        docs,
        so_ky_hieu="126/2014/NĐ-CP",
    )

    nd_82 = find_docs(
        docs,
        so_ky_hieu="82/2020/NĐ-CP",
    )

    ho_tich_2014 = find_docs(
        docs,
        title_contains="Hộ tịch",
        so_ky_hieu="60/2014/QH13",
    )

    # Nhóm 1: Luật HNGĐ 2014
    base_law = law_2014[:1]

    semantic_questions = [
        (
            "Độ tuổi hợp pháp để nam nữ kết hôn là bao nhiêu?",
            "Nam từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên."
        ),
        (
            "Điều kiện kết hôn theo Luật Hôn nhân và gia đình 2014 là gì?",
            "Điều kiện kết hôn gồm: đủ tuổi, tự nguyện, không mất năng lực hành vi dân sự và không thuộc trường hợp cấm kết hôn."
        ),
        (
            "Kết hôn trái pháp luật là gì?",
            "Kết hôn trái pháp luật là việc nam, nữ đã đăng ký kết hôn nhưng một bên hoặc cả hai bên vi phạm điều kiện kết hôn."
        ),
        (
            "Tảo hôn là gì?",
            "Tảo hôn là việc lấy vợ, lấy chồng khi một bên hoặc cả hai bên chưa đủ tuổi kết hôn theo quy định."
        ),
        (
            "Nhà nước có thừa nhận hôn nhân giữa những người cùng giới tính không?",
            "Nhà nước không thừa nhận hôn nhân giữa những người cùng giới tính."
        ),
        (
            "Ai có quyền yêu cầu hủy việc kết hôn trái pháp luật?",
            "Người bị cưỡng ép, lừa dối kết hôn và các cá nhân, cơ quan, tổ chức có thẩm quyền có quyền yêu cầu Tòa án hủy việc kết hôn trái pháp luật."
        ),
        (
            "Chồng có quyền yêu cầu ly hôn khi vợ đang mang thai không?",
            "Chồng không có quyền yêu cầu ly hôn khi vợ đang có thai, sinh con hoặc nuôi con dưới 12 tháng tuổi."
        ),
        (
            "Con dưới 36 tháng tuổi khi ly hôn thường được giao cho ai nuôi?",
            "Con dưới 36 tháng tuổi thường được giao cho mẹ trực tiếp nuôi, trừ trường hợp mẹ không đủ điều kiện hoặc có thỏa thuận khác."
        ),
        (
            "Con từ bao nhiêu tuổi thì Tòa án phải xem xét nguyện vọng khi cha mẹ ly hôn?",
            "Con từ đủ 07 tuổi trở lên thì phải xem xét nguyện vọng của con."
        ),
        (
            "Người không trực tiếp nuôi con sau ly hôn có quyền thăm nom con không?",
            "Người không trực tiếp nuôi con có quyền, nghĩa vụ thăm nom con mà không ai được cản trở."
        ),
        (
            "Tài sản chung của vợ chồng gồm những tài sản nào?",
            "Tài sản chung gồm tài sản do vợ chồng tạo ra, thu nhập do lao động, sản xuất kinh doanh và hoa lợi, lợi tức phát sinh trong thời kỳ hôn nhân."
        ),
        (
            "Tài sản được tặng cho riêng trong thời kỳ hôn nhân là tài sản chung hay riêng?",
            "Tài sản được tặng cho riêng trong thời kỳ hôn nhân là tài sản riêng của người được tặng cho."
        ),
        (
            "Vợ chồng có thể thỏa thuận chế độ tài sản trước khi kết hôn không?",
            "Có. Nếu lựa chọn chế độ tài sản theo thỏa thuận thì thỏa thuận phải lập trước khi kết hôn và được công chứng hoặc chứng thực."
        ),
        (
            "Nghĩa vụ cấp dưỡng được đặt ra trong những trường hợp nào?",
            "Nghĩa vụ cấp dưỡng được đặt ra giữa cha mẹ và con, giữa vợ chồng, anh chị em và các quan hệ gia đình khác theo quy định."
        ),
        (
            "Mức cấp dưỡng do ai quyết định?",
            "Mức cấp dưỡng do các bên thỏa thuận; nếu không thỏa thuận được thì yêu cầu Tòa án giải quyết."
        ),
    ]

    for query, answer in semantic_questions:
        cases.append(make_case(cid, "single-hop-semantic", query, base_law, answer))
        cid += 1

    # Nhóm 2: exact keyword/article questions cho Luật HNGĐ 2014
    article_questions = [
        ("Điều 8 Luật Hôn nhân và gia đình 2014 quy định gì?", "Điều 8 quy định điều kiện kết hôn."),
        ("Điều 10 Luật HNGĐ 2014 quy định ai có quyền yêu cầu hủy kết hôn trái pháp luật?", "Điều 10 quy định người có quyền yêu cầu hủy việc kết hôn trái pháp luật."),
        ("Điều 33 Luật Hôn nhân và gia đình 2014 quy định gì về tài sản chung của vợ chồng?", "Điều 33 quy định về tài sản chung của vợ chồng."),
        ("Điều 43 Luật Hôn nhân và gia đình 2014 quy định gì về tài sản riêng của vợ chồng?", "Điều 43 quy định về tài sản riêng của vợ, chồng."),
        ("Điều 51 Luật HNGĐ 2014 quy định ai có quyền yêu cầu ly hôn?", "Điều 51 quy định quyền yêu cầu giải quyết ly hôn."),
        ("Điều 56 Luật HNGĐ 2014 quy định ly hôn theo yêu cầu của một bên như thế nào?", "Điều 56 quy định căn cứ ly hôn theo yêu cầu của một bên."),
        ("Điều 59 Luật HNGĐ 2014 quy định nguyên tắc chia tài sản khi ly hôn như thế nào?", "Điều 59 quy định nguyên tắc giải quyết tài sản của vợ chồng khi ly hôn."),
        ("Điều 81 Luật HNGĐ 2014 quy định việc trông nom, chăm sóc, nuôi dưỡng con sau ly hôn như thế nào?", "Điều 81 quy định việc trông nom, chăm sóc, nuôi dưỡng, giáo dục con sau ly hôn."),
        ("Điều 82 Luật HNGĐ 2014 quy định nghĩa vụ của người không trực tiếp nuôi con như thế nào?", "Điều 82 quy định nghĩa vụ, quyền của cha, mẹ không trực tiếp nuôi con sau ly hôn."),
        ("Điều 83 Luật HNGĐ 2014 quy định quyền thăm nom con sau ly hôn như thế nào?", "Điều 83 quy định nghĩa vụ, quyền của cha, mẹ trực tiếp nuôi con đối với người không trực tiếp nuôi con."),
        ("Điều 95 Luật HNGĐ 2014 quy định điều kiện mang thai hộ vì mục đích nhân đạo như thế nào?", "Điều 95 quy định điều kiện mang thai hộ vì mục đích nhân đạo."),
        ("Điều 107 Luật HNGĐ 2014 quy định nghĩa vụ cấp dưỡng như thế nào?", "Điều 107 quy định nghĩa vụ cấp dưỡng."),
        ("Điều 110 Luật HNGĐ 2014 quy định nghĩa vụ cấp dưỡng của cha mẹ đối với con như thế nào?", "Điều 110 quy định nghĩa vụ cấp dưỡng của cha, mẹ đối với con."),
        ("Điều 116 Luật HNGĐ 2014 quy định mức cấp dưỡng như thế nào?", "Điều 116 quy định mức cấp dưỡng."),
        ("Điều 117 Luật HNGĐ 2014 quy định phương thức cấp dưỡng như thế nào?", "Điều 117 quy định phương thức cấp dưỡng."),
    ]

    for query, answer in article_questions:
        cases.append(make_case(cid, "exact-keyword-fact", query, base_law, answer))
        cid += 1

    # Nhóm 3: văn bản hướng dẫn
    if guide_2016:
        guide_questions = [
            (
                "Thông tư liên tịch 01/2016 hướng dẫn nội dung nào của Luật Hôn nhân và gia đình?",
                "Thông tư liên tịch 01/2016 hướng dẫn xử lý kết hôn trái pháp luật, thỏa thuận chế độ tài sản vô hiệu và chia tài sản khi ly hôn."
            ),
            (
                "Văn bản nào hướng dẫn xử lý việc kết hôn trái pháp luật?",
                "Thông tư liên tịch 01/2016/TTLT-TANDTC-VKSNDTC-BTP hướng dẫn xử lý việc kết hôn trái pháp luật."
            ),
            (
                "Căn cứ hủy việc kết hôn trái pháp luật được hướng dẫn tại văn bản nào?",
                "Được hướng dẫn tại Thông tư liên tịch 01/2016/TTLT-TANDTC-VKSNDTC-BTP."
            ),
            (
                "Thông tư liên tịch 01/2016 có còn hiệu lực không?",
                "Cần kiểm tra tình trạng hiệu lực trong metadata của văn bản."
            ),
            (
                "Thông tư liên tịch 01/2016 hướng dẫn nguyên tắc giải quyết tài sản của vợ chồng khi ly hôn đúng không?",
                "Có, văn bản này hướng dẫn một số nội dung liên quan đến nguyên tắc giải quyết tài sản khi ly hôn."
            ),
        ]

        for query, answer in guide_questions:
            cases.append(make_case(cid, "exact-keyword-fact", query, guide_2016[:1], answer))
            cid += 1

    # Nhóm 4: Nghị định 126/2014
    if nd_126:
        questions = [
            (
                "Nghị định 126/2014/NĐ-CP quy định chi tiết thi hành luật nào?",
                "Nghị định 126/2014/NĐ-CP quy định chi tiết một số điều và biện pháp thi hành Luật Hôn nhân và gia đình."
            ),
            (
                "Văn bản nào quy định chi tiết một số điều của Luật Hôn nhân và gia đình 2014?",
                "Nghị định 126/2014/NĐ-CP là một trong các văn bản quy định chi tiết thi hành Luật Hôn nhân và gia đình 2014."
            ),
            (
                "Nghị định 126/2014/NĐ-CP có liên quan đến chế độ tài sản của vợ chồng không?",
                "Có, Nghị định này có quy định chi tiết về một số nội dung liên quan đến chế độ tài sản của vợ chồng."
            ),
            (
                "Nghị định 126/2014/NĐ-CP có liên quan đến hôn nhân có yếu tố nước ngoài không?",
                "Có, Nghị định này quy định một số nội dung về quan hệ hôn nhân và gia đình có yếu tố nước ngoài."
            ),
            (
                "Số ký hiệu của nghị định hướng dẫn Luật Hôn nhân và gia đình 2014 là gì?",
                "Một văn bản quan trọng là Nghị định 126/2014/NĐ-CP."
            ),
        ]

        for query, answer in questions:
            cases.append(make_case(cid, "exact-keyword-fact", query, nd_126[:1], answer))
            cid += 1

    # Nhóm 5: Nghị định 82/2020 nếu có trong corpus
    if nd_82:
        questions = [
            (
                "Nghị định 82/2020/NĐ-CP có quy định xử phạt hành chính về hôn nhân gia đình không?",
                "Có, Nghị định 82/2020/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực bổ trợ tư pháp, hành chính tư pháp, hôn nhân và gia đình."
            ),
            (
                "Hành vi tảo hôn có thể bị xử phạt hành chính theo văn bản nào?",
                "Có thể được xử phạt theo Nghị định 82/2020/NĐ-CP nếu thuộc phạm vi điều chỉnh."
            ),
            (
                "Kết hôn giả tạo có thể bị xử phạt theo Nghị định nào?",
                "Có thể bị xử phạt theo Nghị định 82/2020/NĐ-CP."
            ),
            (
                "Nghị định 82/2020/NĐ-CP thuộc loại văn bản gì?",
                "Đây là Nghị định của Chính phủ."
            ),
            (
                "Nghị định 82/2020/NĐ-CP có liên quan đến lĩnh vực hôn nhân gia đình không?",
                "Có, văn bản này có nội dung xử phạt vi phạm hành chính liên quan đến hôn nhân và gia đình."
            ),
        ]

        for query, answer in questions:
            cases.append(make_case(cid, "exact-keyword-fact", query, nd_82[:1], answer))
            cid += 1

    # Nhóm 6: Luật Hộ tịch nếu có
    if ho_tich_2014:
        questions = [
            (
                "Cơ quan nào có thẩm quyền đăng ký kết hôn có yếu tố nước ngoài?",
                "Ủy ban nhân dân cấp huyện có thẩm quyền đăng ký kết hôn có yếu tố nước ngoài theo Luật Hộ tịch."
            ),
            (
                "Luật Hộ tịch 2014 có liên quan đến đăng ký kết hôn không?",
                "Có, Luật Hộ tịch 2014 quy định về đăng ký kết hôn và các việc hộ tịch khác."
            ),
            (
                "Đăng ký kết hôn thuộc phạm vi điều chỉnh của luật nào ngoài Luật Hôn nhân và gia đình?",
                "Đăng ký kết hôn còn thuộc phạm vi điều chỉnh của Luật Hộ tịch."
            ),
            (
                "Việc đăng ký kết hôn có yếu tố nước ngoài do cơ quan nào thực hiện?",
                "Ủy ban nhân dân cấp huyện thực hiện đăng ký kết hôn có yếu tố nước ngoài."
            ),
            (
                "Luật Hộ tịch số 60/2014/QH13 có liên quan đến kết hôn không?",
                "Có, luật này quy định về đăng ký kết hôn trong quản lý hộ tịch."
            ),
        ]

        for query, answer in questions:
            cases.append(make_case(cid, "exact-keyword-fact", query, ho_tich_2014[:1], answer))
            cid += 1

    # Nhóm 7: multi-hop cơ bản chỉ dùng nếu có đủ doc
    if law_2014 and law_2000:
        multi_hop = [
            (
                "Luật Hôn nhân và Gia đình năm 2014 thay thế Luật Hôn nhân và Gia đình năm nào?",
                [law_2014[0], law_2000[0]],
                "Luật Hôn nhân và Gia đình năm 2014 thay thế Luật Hôn nhân và Gia đình năm 2000."
            ),
            (
                "Luật Hôn nhân và Gia đình 2000 hiện còn hiệu lực không nếu đã có Luật Hôn nhân và Gia đình 2014?",
                [law_2000[0], law_2014[0]],
                "Luật Hôn nhân và Gia đình 2000 đã hết hiệu lực toàn bộ sau khi Luật Hôn nhân và Gia đình 2014 có hiệu lực."
            ),
            (
                "Khi trả lời về điều kiện kết hôn hiện hành nên ưu tiên Luật HNGĐ 2014 hay Luật HNGĐ 2000?",
                [law_2014[0], law_2000[0]],
                "Nên ưu tiên Luật Hôn nhân và Gia đình 2014 vì đây là văn bản hiện hành, còn Luật năm 2000 đã hết hiệu lực."
            ),
        ]

        for query, rel_docs, answer in multi_hop:
            cases.append(make_case(cid, "multi-hop-graph", query, rel_docs, answer))
            cid += 1

    # Nhóm 8: tạo thêm câu hỏi theo title để đủ 100
    for doc in docs:
        if cid > 100:
            break

        title = doc["title"]
        so_ky_hieu = doc["so_ky_hieu"]

        if not title:
            continue

        query = f"Văn bản {title} số {so_ky_hieu} quy định nội dung gì liên quan đến hôn nhân gia đình?"
        answer = f"Văn bản {title} số {so_ky_hieu} có nội dung liên quan đến lĩnh vực hôn nhân và gia đình."

        cases.append(
            make_case(
                cid,
                "metadata-title-query",
                query,
                [doc],
                answer,
            )
        )

        cid += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases[:100], f, ensure_ascii=False, indent=2)

    print(f"Saved {len(cases[:100])} test cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()