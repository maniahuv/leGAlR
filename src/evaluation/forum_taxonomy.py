from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def contains_any(text: str, keywords: list[str]) -> bool:
    q = normalize_text(text)
    qa = strip_accents(q)

    for kw in keywords:
        k = normalize_text(kw)
        ka = strip_accents(k)

        if k in q or ka in qa:
            return True

    return False


def count_any(text: str, keywords: list[str]) -> int:
    q = normalize_text(text)
    qa = strip_accents(q)

    count = 0
    for kw in keywords:
        k = normalize_text(kw)
        ka = strip_accents(k)
        if k in q or ka in qa:
            count += 1

    return count


DOMAIN_KEYWORDS = {
    "inheritance_adjacent": [
        "thừa kế",
        "di chúc",
        "di sản",
        "hàng thừa kế",
        "thừa kế thế vị",
        "người thừa kế",
        "truất quyền hưởng di sản",
        "di chúc miệng",
    ],
    "residence_adjacent": [
        "hộ khẩu",
        "nhập hộ khẩu",
        "tách hộ khẩu",
        "thường trú",
        "tạm trú",
        "cư trú",
        "sổ hộ khẩu",
    ],
    "civil_procedure_adjacent": [
        "án phí",
        "án phí sơ thẩm",
        "vắng mặt",
        "ủy quyền tham gia tố tụng",
        "tòa án triệu tập",
        "phiên tòa",
    ],
    "domestic_violence": [
        "bạo lực gia đình",
        "người bị bạo lực",
        "giam cầm",
        "cô lập",
        "xúc phạm danh dự",
        "nhân phẩm",
        "tạm lánh",
        "tố giác bạo lực",
        "tổng đài",
    ],
    "adoption": [
        "nuôi con nuôi",
        "nhận con nuôi",
        "con nuôi",
        "cha mẹ nuôi",
        "mẹ kế nhận con chồng",
        "người nước ngoài nhận con nuôi",
        "hồ sơ nhận nuôi",
    ],
    "civil_status": [
        "hộ tịch",
        "khai sinh",
        "giấy khai sinh",
        "đăng ký khai sinh",
        "giấy chứng nhận kết hôn",
        "đăng ký kết hôn",
        "xác nhận tình trạng hôn nhân",
        "giấy xác nhận độc thân",
        "thay đổi họ",
        "đổi họ",
        "cải chính hộ tịch",
        "bổ sung tên cha",
        "nhận cha con",
        "nhận cha, con",
        "quốc tịch cho con",
        "ghi chú ly hôn",
        "ly hôn ở nước ngoài",
    ],
    "case_law": [
        "án lệ",
        "hội đồng thẩm phán",
        "tòa án nhân dân tối cao",
    ],
}


ISSUE_KEYWORDS = {
    "child_name_change": [
        "thay đổi họ",
        "đổi họ",
        "theo họ mẹ",
        "theo họ cha",
        "họ cha dượng",
        "đổi họ cho con",
    ],
    "birth_registration": [
        "khai sinh",
        "giấy khai sinh",
        "đăng ký khai sinh",
        "làm giấy khai sinh",
        "quốc tịch cho con",
    ],
    "parent_child_recognition": [
        "nhận cha con",
        "nhận cha, con",
        "nhận con ngoài giá thú",
        "bổ sung tên cha",
        "chứng minh quan hệ cha mẹ con",
    ],
    "marriage_registration": [
        "đăng ký kết hôn",
        "giấy chứng nhận kết hôn",
        "tái hôn",
        "giấy xác nhận độc thân",
        "xác nhận tình trạng hôn nhân",
        "nhận giấy đăng ký kết hôn",
    ],
    "marriage_conditions": [
        "điều kiện kết hôn",
        "đủ tuổi kết hôn",
        "tuổi kết hôn",
        "đủ điều kiện kết hôn",
        "kết hôn cần",
        "bao nhiêu tuổi",
    ],
    "prohibited_marriage": [
        "cấm kết hôn",
        "người đang có vợ",
        "người đang có chồng",
        "ba đời",
        "phạm vi 3 đời",
        "phạm vi ba đời",
        "con ruột và con nuôi",
        "sui gia",
    ],
    "adultery_bigamy": [
        "ngoại tình",
        "gian phu",
        "dâm phụ",
        "năm thê bảy thiếp",
        "lắm vợ",
        "người thứ ba",
        "có con ngoài giá thú",
        "đang có vợ",
        "đang có chồng",
    ],
    "mutual_divorce": [
        "thuận tình ly hôn",
        "ly hôn thuận tình",
        "mẫu đơn ly hôn thuận tình",
        "cả hai vợ chồng tự nguyện",
    ],
    "unilateral_divorce": [
        "đơn phương ly hôn",
        "ly hôn đơn phương",
        "yêu cầu ly hôn đơn phương",
        "lý do ly hôn đơn phương",
    ],
    "divorce_general": [
        "ly hôn",
        "ly thân",
        "chấm dứt hôn nhân",
        "sau khi ly hôn",
        "nên ly hôn hay ly thân",
    ],
    "child_custody": [
        "quyền nuôi con",
        "giành quyền nuôi con",
        "người trực tiếp nuôi con",
        "ai được quyền nuôi con",
        "con dưới 36 tháng",
        "con dưới 12 tháng",
        "thăm con",
    ],
    "child_custody_change": [
        "giành lại quyền nuôi con",
        "thay đổi người nuôi con",
        "thay đổi quyền nuôi con",
        "sau 36 tháng",
    ],
    "child_support": [
        "cấp dưỡng nuôi con",
        "tiền cấp dưỡng nuôi con",
        "không chu cấp cho con",
        "trốn cấp dưỡng",
        "mức cấp dưỡng",
    ],
    "spousal_support": [
        "cấp dưỡng cho vợ",
        "cấp dưỡng cho chồng",
        "cấp dưỡng cho người còn lại",
        "cấp dưỡng sau khi ly hôn",
    ],
    "marital_property": [
        "tài sản chung",
        "tài sản riêng",
        "chia tài sản",
        "tài sản khi ly hôn",
        "tiền trúng số",
        "trúng thưởng xổ số",
        "của hồi môn",
        "ly thân",
        "làm dâu",
        "ở rể",
    ],
    "cohabitation_without_registration": [
        "không đăng ký kết hôn",
        "không đkkh",
        "sống chung như vợ chồng",
        "nam nữ sống chung",
        "chung sống nhưng không đăng ký",
    ],
    "parent_child_rights_obligations": [
        "cha mẹ có nghĩa vụ",
        "con có nghĩa vụ",
        "nuôi dưỡng cha mẹ",
        "phụng dưỡng cha mẹ",
        "quyền và nghĩa vụ của con",
        "quyền và nghĩa vụ của cha mẹ",
        "cha mẹ sinh con",
        "con cái",
        "anh chị em",
        "cấp dưỡng giữa anh chị em",
    ],
    "adoption": [
        "nuôi con nuôi",
        "nhận con nuôi",
        "con nuôi",
        "hồ sơ nhận nuôi",
        "người nước ngoài nhận con nuôi",
    ],
    "domestic_violence": [
        "bạo lực gia đình",
        "người bị bạo lực",
        "giam cầm",
        "xúc phạm danh dự",
        "cô lập",
        "tạm lánh",
    ],
    "inheritance": [
        "thừa kế",
        "di chúc",
        "di sản",
        "hàng thừa kế",
        "thừa kế thế vị",
    ],
    "residence_household": [
        "hộ khẩu",
        "nhập hộ khẩu",
        "cư trú",
        "thường trú",
        "tạm trú",
    ],
}


CULTURAL_PHRASE_MARKERS = [
    "là gì",
    "nghĩa là gì",
    "có ý nghĩa gì",
    "tục ngữ",
    "ca dao",
    "thành ngữ",
    "câu nói",
    "câu ca dao",
    "câu tục ngữ",
    "công cha",
    "nghĩa mẹ",
    "năm thê bảy thiếp",
    "có mới nới cũ",
    "gian phu dâm phụ",
    "bách niên giai lão",
    "thuận vợ thuận chồng",
    "cá không ăn muối",
    "đời cha ăn mặn",
    "máu chảy ruột mềm",
    "chén trong sóng",
    "đạo vợ nghĩa chồng",
]


PROCEDURAL_MARKERS = [
    "thủ tục",
    "hồ sơ",
    "giấy tờ",
    "xin giấy",
    "cấp giấy",
    "đăng ký",
    "xác nhận",
    "cải chính",
    "bổ sung",
    "thay đổi",
    "cập nhật",
    "ở đâu",
    "cơ quan nào",
    "thẩm quyền",
    "mẫu đơn",
    "cách điền",
]


DOCUMENT_RELATION_MARKERS = [
    "nghị định nào",
    "thông tư nào",
    "quy định chi tiết",
    "hướng dẫn",
    "sửa đổi",
    "bổ sung nghị định",
    "thay thế",
    "bãi bỏ",
]


MISSING_FACTS_MARKERS = [
    "tôi muốn ly hôn",
    "nên ly hôn hay ly thân",
    "có nên ly hôn",
    "phải làm sao",
    "làm thế nào",
    "tư vấn giúp",
    "xin tư vấn",
]


def classify_legal_domain(text: str) -> tuple[str, str]:
    """
    Return: (legal_domain, scope)
    """
    # Priority: adjacent/out-of-scope first
    if contains_any(text, DOMAIN_KEYWORDS["inheritance_adjacent"]):
        return "inheritance_adjacent", "adjacent"

    if contains_any(text, DOMAIN_KEYWORDS["residence_adjacent"]):
        return "residence_adjacent", "adjacent"

    if contains_any(text, DOMAIN_KEYWORDS["civil_procedure_adjacent"]):
        return "civil_procedure_adjacent", "adjacent"

    if contains_any(text, DOMAIN_KEYWORDS["case_law"]):
        return "case_law", "in_scope"

    if contains_any(text, DOMAIN_KEYWORDS["domestic_violence"]):
        return "domestic_violence", "in_scope"

    if contains_any(text, DOMAIN_KEYWORDS["adoption"]):
        return "adoption", "in_scope"

    if contains_any(text, DOMAIN_KEYWORDS["civil_status"]):
        return "civil_status", "in_scope"

    return "marriage_family_core", "in_scope"


def classify_legal_issue(text: str, legal_domain: str) -> str:
    # Adjacent domains first
    if legal_domain == "inheritance_adjacent":
        return "inheritance"

    if legal_domain == "residence_adjacent":
        return "residence_household"

    if legal_domain == "domestic_violence":
        return "domestic_violence"

    if legal_domain == "adoption":
        return "adoption"

    # Specific issues in priority order
    priority = [
        "child_name_change",
        "birth_registration",
        "parent_child_recognition",
        "marriage_registration",
        "marriage_conditions",
        "prohibited_marriage",
        "adultery_bigamy",
        "mutual_divorce",
        "unilateral_divorce",
        "child_custody_change",
        "child_custody",
        "child_support",
        "spousal_support",
        "marital_property",
        "cohabitation_without_registration",
        "parent_child_rights_obligations",
        "divorce_general",
    ]

    for issue in priority:
        if contains_any(text, ISSUE_KEYWORDS.get(issue, [])):
            return issue

    if legal_domain == "civil_status":
        return "civil_status_general"

    if legal_domain == "case_law":
        return "case_law"

    return "other"


def classify_technical_challenge(
    text: str,
    legal_domain: str,
    legal_issue: str,
    scope: str,
) -> str:
    if scope != "in_scope":
        return "out_of_scope_detection"

    if contains_any(text, DOCUMENT_RELATION_MARKERS):
        return "document_relation"

    if legal_domain == "case_law" or legal_issue == "case_law":
        return "case_application"

    if contains_any(text, MISSING_FACTS_MARKERS):
        return "missing_facts"

    if contains_any(text, PROCEDURAL_MARKERS):
        return "procedural_multi_authority"

    if contains_any(text, CULTURAL_PHRASE_MARKERS):
        return "lexical_gap_cultural_phrase"

    # Multi-authority usually needs several legal bases
    if legal_issue in {
        "marital_property",
        "child_support",
        "spousal_support",
        "cohabitation_without_registration",
        "child_custody",
        "child_custody_change",
        "adultery_bigamy",
        "parent_child_recognition",
    }:
        return "multi_authority"

    # Lexical gap but not necessarily cultural phrase
    if contains_any(
        text,
        [
            "ngoại tình tư tưởng",
            "gian phu",
            "dâm phụ",
            "có mới nới cũ",
            "đòi con",
            "giành con",
            "chia tay",
            "hủy hôn",
            "không chu cấp",
            "trốn cấp dưỡng",
        ],
    ):
        return "lexical_gap"

    return "direct_lookup"


def classify_record(record: dict[str, Any]) -> dict[str, str]:
    text = " ".join(
        str(record.get(k) or "")
        for k in ["benchmark_query", "classification_text", "title", "question_text"]
    )

    legal_domain, scope = classify_legal_domain(text)
    legal_issue = classify_legal_issue(text, legal_domain)
    technical_challenge = classify_technical_challenge(
        text=text,
        legal_domain=legal_domain,
        legal_issue=legal_issue,
        scope=scope,
    )

    return {
        "legal_domain": legal_domain,
        "legal_issue": legal_issue,
        "technical_challenge": technical_challenge,
        "scope": scope,
    }


LABEL_SCHEMA = {
    "legal_domain": [
        "marriage_family_core",
        "civil_status",
        "adoption",
        "domestic_violence",
        "case_law",
        "inheritance_adjacent",
        "residence_adjacent",
        "civil_procedure_adjacent",
        "other_out_of_scope",
    ],
    "legal_issue": sorted(set(ISSUE_KEYWORDS.keys()) | {"civil_status_general", "case_law", "other"}),
    "technical_challenge": [
        "direct_lookup",
        "lexical_gap",
        "lexical_gap_cultural_phrase",
        "multi_authority",
        "procedural_multi_authority",
        "document_relation",
        "case_application",
        "missing_facts",
        "out_of_scope_detection",
    ],
    "scope": [
        "in_scope",
        "adjacent",
        "out_of_scope",
    ],
}