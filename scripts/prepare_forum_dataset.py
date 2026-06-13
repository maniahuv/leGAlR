from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.forum_taxonomy import LABEL_SCHEMA, classify_record


BOILERPLATE_PATTERNS = [
    r"trang chủ\s+thư viện pháp luật",
    r"chủ quản:\s*công ty\s+thư viện pháp luật",
    r"danh sách diễn đàn",
    r"đã có toàn văn luật",
    r"thư viện pháp luật",
    r"vui lòng đăng nhập",
    r"bản quyền thuộc",
    r"cộng đồng dân luật",
]


ANSWER_LEAK_PATTERNS = [
    r"\bcăn cứ\b",
    r"\btheo điều\b",
    r"\btại điều\b",
    r"\bkhoản\s+\d+\s+điều\b",
    r"\bluật\s+[a-zà-ỹ\s]+20\d{2}\b",
    r"\bnghị định\s+\d+",
    r"\bthông tư\s+\d+",
    r"\bbộ luật\b",
    r"\bquy định như sau\b",
]


SPAM_PATTERNS = [
    r"https?://(?!thuvienphapluat\.vn)",
    r"\bcasino\b",
    r"\bbet\b",
    r"\bviral\b",
    r"\bthesport\b",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse error at line {line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")

            obj["_raw_line_no"] = line_no
            records.append(obj)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    text = text or ""
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^\w\sà-ỹđ]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_url(url: str) -> str:
    url = normalize_space(url)

    if not url:
        return ""

    parts = urlsplit(url)

    # Remove fragment, keep query if any
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def has_fragment_url(url: str) -> bool:
    return bool(urlsplit(url or "").fragment)


def contains_pattern(text: str, patterns: list[str]) -> bool:
    t = normalize_space(text).lower()

    for pattern in patterns:
        if re.search(pattern, t, flags=re.IGNORECASE):
            return True

    return False


def strip_boilerplate(text: str) -> str:
    text = normalize_space(text)

    if not text:
        return ""

    # Cut at common boilerplate markers
    lowered = text.lower()
    cut_positions = []

    for pattern in BOILERPLATE_PATTERNS:
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        text = text[: min(cut_positions)].strip()

    return normalize_space(text)


def remove_answer_leak_from_excerpt(excerpt: str) -> str:
    """
    Excerpt thường chứa luôn phần giải đáp: "Căn cứ Điều..."
    Để phân loại thì chỉ cần ngữ cảnh tình huống, nên cắt trước các marker căn cứ pháp lý.
    """
    excerpt = strip_boilerplate(excerpt)

    if not excerpt:
        return ""

    lowered = excerpt.lower()
    cut_positions = []

    for pattern in ANSWER_LEAK_PATTERNS:
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        excerpt = excerpt[: min(cut_positions)].strip()

    return normalize_space(excerpt)


def choose_benchmark_query(record: dict[str, Any]) -> str:
    """
    Query benchmark nên giống câu người dùng nhập.
    Không dùng full_question/excerpt dài vì có thể chứa boilerplate hoặc căn cứ pháp lý.
    """
    title = normalize_space(record.get("title") or "")
    question_text = strip_boilerplate(record.get("question_text") or "")

    # Nếu question_text rỗng hoặc quá dài/nhiễu, dùng title
    if not question_text:
        return title

    if contains_pattern(question_text, BOILERPLATE_PATTERNS):
        return title

    if len(question_text) > 250:
        return title

    # Nếu question_text chỉ lặp lại title hoặc title rõ hơn, dùng title
    if title and normalize_for_key(question_text).startswith(normalize_for_key(title)):
        return title

    return question_text or title


def build_classification_text(record: dict[str, Any]) -> str:
    title = normalize_space(record.get("title") or "")
    question_text = strip_boilerplate(record.get("question_text") or "")
    excerpt = remove_answer_leak_from_excerpt(record.get("excerpt") or "")

    parts = []

    for part in [title, question_text, excerpt]:
        if part and normalize_for_key(part) not in {normalize_for_key(p) for p in parts}:
            parts.append(part)

    text = ". ".join(parts)
    text = normalize_space(text)

    # Giới hạn để tránh kéo nhiễu
    if len(text) > 900:
        text = text[:900].rstrip() + "…"

    return text


def detect_quality_flags(record: dict[str, Any]) -> list[str]:
    flags: list[str] = []

    title = normalize_space(record.get("title") or "")
    question_text = normalize_space(record.get("question_text") or "")
    full_question = normalize_space(record.get("full_question") or "")
    excerpt = normalize_space(record.get("excerpt") or "")
    url = normalize_space(record.get("url") or "")

    combined = " ".join([title, question_text, full_question, excerpt])

    if has_fragment_url(url):
        flags.append("comment_or_reply_url")

    if contains_pattern(combined, BOILERPLATE_PATTERNS):
        flags.append("has_boilerplate")

    if contains_pattern(combined, SPAM_PATTERNS):
        flags.append("possible_spam")

    if len(title) < 15:
        flags.append("short_title")

    if not title:
        flags.append("missing_title")

    if not excerpt:
        flags.append("missing_excerpt")

    if "..." in excerpt or "&..." in excerpt:
        flags.append("possibly_truncated_excerpt")

    return sorted(set(flags))


def record_type(record: dict[str, Any]) -> str:
    flags = set(record.get("quality_flags") or [])

    if "possible_spam" in flags:
        return "noise"

    if "comment_or_reply_url" in flags:
        return "comment_or_reply"

    return "thread"


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)

    url = normalize_space(record.get("url") or "")

    enriched["canonical_url"] = canonicalize_url(url)
    enriched["title_clean"] = normalize_space(record.get("title") or "")
    enriched["excerpt_clean"] = remove_answer_leak_from_excerpt(record.get("excerpt") or "")
    enriched["question_text_clean"] = strip_boilerplate(record.get("question_text") or "")
    enriched["benchmark_query"] = choose_benchmark_query(record)
    enriched["classification_text"] = build_classification_text(record)
    enriched["quality_flags"] = detect_quality_flags(record)
    enriched["record_type"] = record_type(enriched)

    labels = classify_record(enriched)
    enriched.update(labels)

    return enriched


def dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Dedup theo canonical_url, fallback theo title normalized.
    Ưu tiên giữ thread gốc thay vì comment/reply.
    """
    best_by_key: dict[str, dict[str, Any]] = {}

    for record in records:
        url_key = record.get("canonical_url") or ""
        title_key = normalize_for_key(record.get("title_clean") or record.get("title") or "")
        key = url_key or f"title:{title_key}"

        if key not in best_by_key:
            best_by_key[key] = record
            continue

        old = best_by_key[key]

        def score(r: dict[str, Any]) -> int:
            s = 0
            if r.get("record_type") == "thread":
                s += 10
            if r.get("excerpt_clean"):
                s += 3
            if r.get("question_text_clean"):
                s += 2
            if not r.get("quality_flags"):
                s += 1
            return s

        if score(record) > score(old):
            best_by_key[key] = record

    return list(best_by_key.values())


def write_manual_review_sample(
    path: Path,
    records: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Lấy mẫu có cân bằng tương đối theo domain/challenge
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for r in records:
        key = (r.get("legal_domain", "unknown"), r.get("technical_challenge", "unknown"))
        buckets.setdefault(key, []).append(r)

    sampled: list[dict[str, Any]] = []

    per_bucket = max(1, sample_size // max(1, len(buckets)))

    for bucket_records in buckets.values():
        random.shuffle(bucket_records)
        sampled.extend(bucket_records[:per_bucket])

    if len(sampled) < sample_size:
        remaining = [r for r in records if r not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[: sample_size - len(sampled)])

    sampled = sampled[:sample_size]

    columns = [
        "_raw_line_no",
        "thread_id",
        "title_clean",
        "benchmark_query",
        "classification_text",
        "url",
        "record_type",
        "quality_flags",
        "legal_domain",
        "legal_issue",
        "technical_challenge",
        "scope",
        "review_legal_domain",
        "review_legal_issue",
        "review_technical_challenge",
        "review_scope",
        "review_note",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for r in sampled:
            row = {c: r.get(c, "") for c in columns}
            row["quality_flags"] = ";".join(r.get("quality_flags") or [])
            writer.writerow(row)


def write_label_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(LABEL_SCHEMA, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary(path: Path, raw_records: list[dict[str, Any]], deduped: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def counter_table(counter: Counter, title: str) -> list[str]:
        lines = []
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Label | Count | Ratio |")
        lines.append("|---|---:|---:|")

        total = sum(counter.values()) or 1

        for key, count in counter.most_common():
            ratio = count / total * 100
            lines.append(f"| {key} | {count} | {ratio:.1f}% |")

        lines.append("")
        return lines

    scope_counter = Counter(r.get("scope", "unknown") for r in deduped)
    domain_counter = Counter(r.get("legal_domain", "unknown") for r in deduped)
    issue_counter = Counter(r.get("legal_issue", "unknown") for r in deduped)
    challenge_counter = Counter(r.get("technical_challenge", "unknown") for r in deduped)
    record_type_counter = Counter(r.get("record_type", "unknown") for r in deduped)

    flag_counter: Counter[str] = Counter()
    for r in deduped:
        for flag in r.get("quality_flags") or []:
            flag_counter[flag] += 1

    lines = [
        "# Danluat Hôn nhân & Gia đình - Step 3 Summary",
        "",
        f"- Raw records: {len(raw_records)}",
        f"- Deduped records: {len(deduped)}",
        f"- Dropped duplicates: {len(raw_records) - len(deduped)}",
        "",
    ]

    lines += counter_table(record_type_counter, "Record type")
    lines += counter_table(scope_counter, "Scope")
    lines += counter_table(domain_counter, "Legal domain")
    lines += counter_table(issue_counter, "Legal issue")
    lines += counter_table(challenge_counter, "Technical challenge")
    lines += counter_table(flag_counter, "Quality flags")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `benchmark_query` ưu tiên lấy từ title/question_text ngắn, không dùng excerpt dài để tránh rò rỉ căn cứ pháp lý."
    )
    lines.append(
        "- `classification_text` dùng title + question_text + excerpt đã lọc để phân loại chính xác hơn."
    )
    lines.append(
        "- Nhãn hiện tại là rule-based preliminary labels, cần review tay trước khi coi là gold label."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and preliminarily label Danluat Hôn nhân & Gia đình forum dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw JSONL file crawled from Danluat forum.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/evaluation/forum_danluat_168",
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120,
        help="Manual review CSV sample size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    raw_records = read_jsonl(input_path)
    all_labeled = [enrich_record(r) for r in raw_records]
    deduped = dedup_records(all_labeled)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "danluat_168_all_labeled.jsonl"
    dedup_path = out_dir / "danluat_168_clean_labeled_dedup.jsonl"
    sample_path = out_dir / "danluat_168_manual_review_sample.csv"
    summary_path = out_dir / "danluat_168_summary.md"
    schema_path = out_dir / "danluat_168_label_schema.json"

    write_jsonl(all_path, all_labeled)
    write_jsonl(dedup_path, deduped)
    write_manual_review_sample(sample_path, deduped, args.sample_size, args.seed)
    write_summary(summary_path, raw_records, deduped)
    write_label_schema(schema_path)

    print("Done.")
    print(f"Raw records:      {len(raw_records)}")
    print(f"All labeled:      {all_path}")
    print(f"Deduped records:  {len(deduped)}")
    print(f"Dedup output:     {dedup_path}")
    print(f"Manual sample:    {sample_path}")
    print(f"Summary:          {summary_path}")
    print(f"Label schema:     {schema_path}")


if __name__ == "__main__":
    main()