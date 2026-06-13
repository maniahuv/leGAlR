from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    text = text.replace("đkkh", "đăng ký kết hôn")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def contains_any(text: str, keywords: list[str]) -> bool:
    q = normalize_text(text)
    qa = strip_accents(q)
    for kw in keywords:
        k = normalize_text(kw)
        ka = strip_accents(k)
        if k and (k in q or ka in qa):
            return True
    return False


def contains_word(text: str, word: str) -> bool:
    q = strip_accents(normalize_text(text))
    w = strip_accents(normalize_text(word))
    return re.search(rf"(?<!\w){re.escape(w)}(?!\w)", q) is not None


OUT_OF_SCOPE_KEYWORDS = [
    "kt3", "chuyển đổi kt3", "chuyen doi kt3",
    "nhắn tin chuyện phòng the", "nhan tin chuyen phong the", "phòng the",
    "chuyện tình cảm phức tạp", "chuyen tinh cam phuc tap",
    "đủ 18 tuổi", "du 18 tuoi", "tính là 18 tuổi", "tinh la 18 tuoi",
    "luật đường bộ", "giày đá bóng", "casino", "viral",
]

INHERITANCE_KEYWORDS = [
    "thừa kế", "thua ke", "di chúc", "di chuc", "di sản", "di san",
    "hàng thừa kế", "hang thua ke", "thừa kế thế vị", "thua ke the vi",
    "truất quyền hưởng di sản", "truat quyen huong di san",
    "di chúc miệng", "di chuc mieng",
]

RESIDENCE_KEYWORDS = [
    "hộ khẩu", "ho khau", "nhập hộ khẩu", "nhap ho khau", "tách hộ khẩu", "tach ho khau",
    "thường trú", "thuong tru", "tạm trú", "tam tru", "cư trú", "cu tru", "sổ hộ khẩu", "so ho khau",
]

CIVIL_PROCEDURE_KEYWORDS = [
    "án phí", "an phi", "án phí sơ thẩm", "an phi so tham", "vắng mặt", "vang mat",
    "ủy quyền tham gia tố tụng", "uy quyen tham gia to tung", "triệu tập", "trieu tap",
]

DOMESTIC_VIOLENCE_KEYWORDS = [
    "bạo lực gia đình", "bao luc gia dinh", "bạo hành", "bao hanh", "đánh đập", "danh dap",
    "giam cầm", "giam cam", "cô lập", "co lap", "xúc phạm danh dự", "xuc pham danh du",
    "nhân phẩm", "nhan pham", "tạm lánh", "tam lanh", "tố giác bạo lực", "to giac bao luc",
]

ADOPTION_STRONG_KEYWORDS = [
    "nuôi con nuôi", "nuoi con nuoi", "nhận con nuôi", "nhan con nuoi", "con nuôi", "con nuoi",
    "cha mẹ nuôi", "cha me nuoi", "mẹ kế nhận con chồng", "me ke nhan con chong",
    "người nước ngoài nhận con nuôi", "nguoi nuoc ngoai nhan con nuoi", "hồ sơ nhận nuôi", "ho so nhan nuoi",
]

CIVIL_STATUS_KEYWORDS = [
    "hộ tịch", "ho tich", "khai sinh", "giấy khai sinh", "giay khai sinh", "đăng ký khai sinh", "dang ky khai sinh",
    "giấy chứng nhận kết hôn", "giay chung nhan ket hon", "đăng ký kết hôn", "dang ky ket hon",
    "xác nhận tình trạng hôn nhân", "xac nhan tinh trang hon nhan", "giấy xác nhận độc thân", "giay xac nhan doc than",
    "thay đổi họ", "thay doi ho", "đổi họ", "doi ho", "cải chính hộ tịch", "cai chinh ho tich",
    "bổ sung tên cha", "bo sung ten cha", "nhận cha con", "nhan cha con", "nhận cha, con", "nhan cha, con",
    "quốc tịch cho con", "quoc tich cho con", "ghi chú ly hôn", "ghi chu ly hon", "ly hôn ở nước ngoài", "ly hon o nuoc ngoai",
]

MARRIAGE_CORE_KEYWORDS = [
    "ly hôn", "ly hon", "ly thân", "ly than", "quyền nuôi con", "quyen nuoi con", "giành quyền nuôi con", "gianh quyen nuoi con",
    "cấp dưỡng", "cap duong", "chu cấp", "chu cap", "tài sản chung", "tai san chung", "tài sản riêng", "tai san rieng",
    "chia tài sản", "chia tai san", "ngoại tình", "ngoai tinh", "kết hôn", "ket hon", "vợ chồng", "vo chong",
    "sống chung như vợ chồng", "song chung nhu vo chong", "không đăng ký kết hôn", "khong dang ky ket hon",
]

PROCEDURAL_MARKERS = [
    "thủ tục", "thu tuc", "hồ sơ", "ho so", "giấy tờ", "giay to", "xin giấy", "xin giay", "cấp giấy", "cap giay",
    "đăng ký", "dang ky", "xác nhận", "xac nhan", "cải chính", "cai chinh", "bổ sung", "bo sung", "thay đổi", "thay doi",
    "cập nhật", "cap nhat", "ở đâu", "o dau", "cơ quan nào", "co quan nao", "thẩm quyền", "tham quyen", "mẫu đơn", "mau don", "cách điền", "cach dien",
]

DOCUMENT_RELATION_MARKERS = [
    "nghị định nào", "nghi dinh nao", "thông tư nào", "thong tu nao", "quy định chi tiết", "quy dinh chi tiet",
    "văn bản nào hướng dẫn", "van ban nao huong dan", "sửa đổi", "sua doi", "thay thế", "thay the", "bãi bỏ", "bai bo",
]

MISSING_FACTS_MARKERS = [
    "tôi muốn ly hôn", "toi muon ly hon", "nên ly hôn hay ly thân", "nen ly hon hay ly than",
    "có nên ly hôn", "co nen ly hon", "phải làm sao", "phai lam sao", "tư vấn giúp", "tu van giup", "xin tư vấn", "xin tu van",
]

CULTURAL_PHRASE_MARKERS = [
    "là gì", "la gi", "nghĩa là gì", "nghia la gi", "có ý nghĩa gì", "co y nghia gi", "tục ngữ", "tuc ngu", "ca dao", "thành ngữ", "thanh ngu",
    "công cha", "cong cha", "nghĩa mẹ", "nghia me", "năm thê bảy thiếp", "nam the bay thiep", "có mới nới cũ", "co moi noi cu",
    "gian phu dâm phụ", "gian phu dam phu", "bách niên giai lão", "bach nien giai lao", "thuận vợ thuận chồng", "thuan vo thuan chong",
    "cá không ăn muối", "ca khong an muoi", "đời cha ăn mặn", "doi cha an man", "máu chảy ruột mềm", "mau chay ruot mem",
]

ISSUE_PRIORITY = [
    "cohabitation_without_registration",
    "child_support",
    "spousal_support",
    "child_custody_change",
    "child_custody",
    "parent_child_recognition",
    "child_name_change",
    "birth_registration",
    "unilateral_divorce",
    "mutual_divorce",
    "divorce_general",
    "marital_property",
    "adultery_bigamy",
    "prohibited_marriage",
    "marriage_conditions",
    "marriage_registration",
    "parent_child_rights_obligations",
]

ISSUE_KEYWORDS = {
    "cohabitation_without_registration": ["không đăng ký kết hôn", "khong dang ky ket hon", "chưa đăng ký kết hôn", "chua dang ky ket hon", "không đkkh", "khong dkkh", "chưa đkkh", "chua dkkh", "sống chung như vợ chồng", "song chung nhu vo chong", "chung sống như vợ chồng", "chung song nhu vo chong", "ở chung mà chưa đăng ký", "o chung ma chua dang ky"],
    "child_support": ["cấp dưỡng nuôi con", "cap duong nuoi con", "tiền cấp dưỡng nuôi con", "tien cap duong nuoi con", "trợ cấp nuôi con", "tro cap nuoi con", "trợ cấp cho con", "tro cap cho con", "chu cấp cho con", "chu cap cho con", "không chu cấp", "khong chu cap", "trốn cấp dưỡng", "tron cap duong", "mức cấp dưỡng", "muc cap duong"],
    "spousal_support": ["cấp dưỡng cho vợ", "cap duong cho vo", "cấp dưỡng cho chồng", "cap duong cho chong", "cấp dưỡng sau khi ly hôn", "cap duong sau khi ly hon"],
    "child_custody_change": ["giành lại quyền nuôi con", "gianh lai quyen nuoi con", "thay đổi người nuôi con", "thay doi nguoi nuoi con", "thay đổi quyền nuôi con", "thay doi quyen nuoi con", "sau 36 tháng", "sau 36 thang"],
    "child_custody": ["quyền nuôi con", "quyen nuoi con", "giành quyền nuôi con", "gianh quyen nuoi con", "giành con", "gianh con", "nuôi con sau ly hôn", "nuoi con sau ly hon", "người trực tiếp nuôi con", "nguoi truc tiep nuoi con", "ai được quyền nuôi con", "ai duoc quyen nuoi con", "chăm sóc con", "cham soc con", "thăm con", "tham con"],
    "parent_child_recognition": ["nhận cha con", "nhan cha con", "nhận cha, con", "nhan cha, con", "nhận con ngoài giá thú", "nhan con ngoai gia thu", "bổ sung tên cha", "bo sung ten cha", "chứng minh quan hệ cha mẹ con", "chung minh quan he cha me con"],
    "child_name_change": ["thay đổi họ", "thay doi ho", "đổi họ", "doi ho", "theo họ mẹ", "theo ho me", "theo họ cha", "theo ho cha", "họ cha dượng", "ho cha duong", "đổi họ cho con", "doi ho cho con"],
    "birth_registration": ["khai sinh", "giấy khai sinh", "giay khai sinh", "đăng ký khai sinh", "dang ky khai sinh", "làm giấy khai sinh", "lam giay khai sinh", "quốc tịch cho con", "quoc tich cho con"],
    "unilateral_divorce": ["đơn phương ly hôn", "don phuong ly hon", "ly hôn đơn phương", "ly hon don phuong", "li hôn đơn phương", "li hon don phuong", "yêu cầu ly hôn đơn phương", "yeu cau ly hon don phuong", "không cần bằng chứng", "khong can bang chung", "hủy hôn do vợ bỏ", "huy hon do vo bo"],
    "mutual_divorce": ["thuận tình ly hôn", "thuan tinh ly hon", "ly hôn thuận tình", "ly hon thuan tinh", "mẫu đơn ly hôn thuận tình", "mau don ly hon thuan tinh"],
    "divorce_general": ["ly hôn", "ly hon", "li hôn", "li hon", "ly thân", "ly than", "chấm dứt hôn nhân", "cham dut hon nhan", "hủy hôn", "huy hon"],
    "marital_property": ["tài sản chung", "tai san chung", "tài sản riêng", "tai san rieng", "xác định tài sản", "xac dinh tai san", "chia tài sản", "chia tai san", "phân chia tài sản", "phan chia tai san", "tài sản khi ly hôn", "tai san khi ly hon", "tiền trúng số", "tien trung so", "của hồi môn", "cua hoi mon", "làm dâu", "lam dau", "ở rể", "o re", "giữ nhà", "giu nha"],
    "adultery_bigamy": ["ngoại tình", "ngoai tinh", "gian phu", "dâm phụ", "dam phu", "năm thê bảy thiếp", "nam the bay thiep", "lắm vợ", "lam vo", "người thứ ba", "nguoi thu ba", "có con ngoài giá thú", "co con ngoai gia thu", "đang có vợ", "dang co vo", "đang có chồng", "dang co chong"],
    "prohibited_marriage": ["cấm kết hôn", "cam ket hon", "người đang có vợ", "nguoi dang co vo", "người đang có chồng", "nguoi dang co chong", "ba đời", "ba doi", "3 đời", "3 doi", "phạm vi 3 đời", "pham vi 3 doi", "phạm vi ba đời", "pham vi ba doi", "con ruột và con nuôi", "con ruot va con nuoi", "sui gia"],
    "marriage_conditions": ["điều kiện kết hôn", "dieu kien ket hon", "đủ tuổi kết hôn", "du tuoi ket hon", "tuổi kết hôn", "tuoi ket hon", "đủ điều kiện kết hôn", "du dieu kien ket hon", "muốn kết hôn", "muon ket hon", "kết hôn với", "ket hon voi", "kết hôn giữa", "ket hon giua"],
    "marriage_registration": ["đăng ký kết hôn", "dang ky ket hon", "giấy chứng nhận kết hôn", "giay chung nhan ket hon", "tái hôn", "tai hon", "giấy xác nhận độc thân", "giay xac nhan doc than", "xác nhận tình trạng hôn nhân", "xac nhan tinh trang hon nhan", "lấy vợ việt nam", "lay vo viet nam", "kết hôn với người nước ngoài", "ket hon voi nguoi nuoc ngoai"],
    "parent_child_rights_obligations": ["cha mẹ có nghĩa vụ", "cha me co nghia vu", "con có nghĩa vụ", "con co nghia vu", "nuôi dưỡng cha mẹ", "nuoi duong cha me", "phụng dưỡng cha mẹ", "phung duong cha me", "quyền và nghĩa vụ của con", "quyen va nghia vu cua con", "quyền và nghĩa vụ của cha mẹ", "quyen va nghia vu cua cha me", "cha mẹ sinh con", "cha me sinh con", "anh chị em", "anh chi em"],
}


def _best_text(record: dict[str, Any]) -> str:
    # V2 cố tình tránh raw full_question/question_text dài vì dễ dính boilerplate.
    parts = [
        record.get("benchmark_query"),
        record.get("title_clean"),
        record.get("title"),
        record.get("excerpt_clean"),
        record.get("classification_text"),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = str(p or "").strip()
        k = normalize_text(p)
        if p and k not in seen:
            seen.add(k)
            out.append(p)
    return ". ".join(out)


def classify_legal_domain(text: str) -> tuple[str, str]:
    t = text or ""

    if contains_any(t, OUT_OF_SCOPE_KEYWORDS):
        return "other_out_of_scope", "out_of_scope"

    # Domain-specific, high-confidence categories first.
    if contains_any(t, ADOPTION_STRONG_KEYWORDS):
        # Nếu câu hỏi thật sự là điều kiện kết hôn giữa con nuôi/con đẻ thì vẫn thuộc HN&GĐ lõi.
        if contains_any(t, ["kết hôn", "ket hon", "có được kết hôn", "co duoc ket hon"]):
            return "marriage_family_core", "in_scope"
        return "adoption", "in_scope"

    # Nếu title/query tập trung vào quyền nuôi con/cấp dưỡng thì không để bạo lực kéo domain.
    if contains_any(t, ISSUE_KEYWORDS["child_custody"] + ISSUE_KEYWORDS["child_support"]):
        return "marriage_family_core", "in_scope"

    if contains_any(t, DOMESTIC_VIOLENCE_KEYWORDS):
        return "domestic_violence", "in_scope"

    # Ly hôn/kết hôn/tài sản là lõi, ưu tiên trước civil_status nếu có cả hai.
    if contains_any(t, ["ly hôn", "ly hon", "li hôn", "li hon", "ly thân", "ly than", "tài sản", "tai san", "quyền nuôi con", "quyen nuoi con", "cấp dưỡng", "cap duong", "ngoại tình", "ngoai tinh", "sống chung như vợ chồng", "song chung nhu vo chong", "không đăng ký kết hôn", "khong dang ky ket hon"]):
        return "marriage_family_core", "in_scope"

    if contains_any(t, CIVIL_STATUS_KEYWORDS):
        return "civil_status", "in_scope"

    if contains_any(t, INHERITANCE_KEYWORDS):
        return "inheritance_adjacent", "adjacent"

    if contains_any(t, RESIDENCE_KEYWORDS):
        return "residence_adjacent", "adjacent"

    if contains_any(t, CIVIL_PROCEDURE_KEYWORDS):
        return "civil_procedure_adjacent", "adjacent"

    return "marriage_family_core", "in_scope"


def classify_legal_issue(text: str, legal_domain: str) -> str:
    t = text or ""

    if legal_domain == "other_out_of_scope":
        return "out_of_scope"
    if legal_domain == "inheritance_adjacent":
        return "inheritance"
    if legal_domain == "residence_adjacent":
        return "residence_household"
    if legal_domain == "domestic_violence":
        # Nếu trọng tâm là custody/support dù có bạo lực trong context, giữ issue con/cấp dưỡng.
        for issue in ["child_support", "child_custody", "adultery_bigamy"]:
            if contains_any(t, ISSUE_KEYWORDS[issue]):
                return issue
        return "domestic_violence"
    if legal_domain == "adoption":
        return "adoption"

    for issue in ISSUE_PRIORITY:
        if contains_any(t, ISSUE_KEYWORDS.get(issue, [])):
            return issue

    if legal_domain == "civil_status":
        return "civil_status_general"

    return "other"


def classify_technical_challenge(text: str, legal_domain: str, legal_issue: str, scope: str) -> str:
    t = text or ""

    if scope == "out_of_scope":
        return "out_of_scope_detection"

    if contains_any(t, DOCUMENT_RELATION_MARKERS):
        return "document_relation"

    if legal_domain == "case_law" or legal_issue == "case_law":
        return "case_application"

    # Procedure phải ưu tiên hơn missing_facts vì data forum có rất nhiều câu hỏi thủ tục.
    if contains_any(t, PROCEDURAL_MARKERS):
        return "procedural_multi_authority"

    if contains_any(t, MISSING_FACTS_MARKERS):
        return "missing_facts"

    if contains_any(t, CULTURAL_PHRASE_MARKERS):
        return "lexical_gap_cultural_phrase"

    if legal_issue in {
        "marital_property", "child_support", "spousal_support", "cohabitation_without_registration",
        "child_custody", "child_custody_change", "adultery_bigamy", "parent_child_recognition",
        "marriage_conditions", "prohibited_marriage",
    }:
        return "multi_authority"

    if contains_any(t, ["đòi con", "doi con", "giành con", "gianh con", "chia tay", "hủy hôn", "huy hon", "ngoại tình tư tưởng", "ngoai tinh tu tuong", "gian phu", "dâm phụ", "dam phu", "không chu cấp", "khong chu cap", "li hôn", "li hon", "muon nuoi", "muốn nuôi"]):
        return "lexical_gap"

    return "direct_lookup"


def classify_record(record: dict[str, Any]) -> dict[str, str]:
    text = _best_text(record)
    legal_domain, scope = classify_legal_domain(text)
    legal_issue = classify_legal_issue(text, legal_domain)
    technical_challenge = classify_technical_challenge(text, legal_domain, legal_issue, scope)
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
    "legal_issue": sorted(set(ISSUE_KEYWORDS) | {"adoption", "domestic_violence", "inheritance", "residence_household", "civil_status_general", "out_of_scope", "other"}),
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
    "scope": ["in_scope", "adjacent", "out_of_scope"],
}
