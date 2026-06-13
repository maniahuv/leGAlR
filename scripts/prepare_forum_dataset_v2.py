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

try:
    from src.evaluation.forum_taxonomy_v2 import LABEL_SCHEMA, classify_record
except ModuleNotFoundError:
    from forum_taxonomy_v2 import LABEL_SCHEMA, classify_record


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
    r"\bcăn cứ\b", r"\btheo điều\b", r"\btại điều\b", r"\bkhoản\s+\d+\s+điều\b",
    r"\bluật\s+[a-zà-ỹ\s]+20\d{2}\b", r"\bnghị định\s+\d+", r"\bthông tư\s+\d+", r"\bbộ luật\b", r"\bquy định như sau\b",
]

SPAM_PATTERNS = [r"https?://(?!thuvienphapluat\.vn)", r"\bcasino\b", r"\bviral\b", r"\bthesport\b"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse error at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not an object")
            obj["_raw_line_no"] = line_no
            records.append(obj)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    text = html.unescape(text or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^\w\sà-ỹđ]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    url = normalize_space(url)
    if not url:
        return ""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, p.query, ""))


def has_fragment_url(url: str) -> bool:
    return bool(urlsplit(url or "").fragment)


def contains_pattern(text: str, patterns: list[str]) -> bool:
    t = normalize_space(text).lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def strip_boilerplate(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return ""
    lowered = text.lower()
    cut_positions = []
    for p in BOILERPLATE_PATTERNS:
        m = re.search(p, lowered, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())
    if cut_positions:
        text = text[: min(cut_positions)].strip()
    return normalize_space(text)


def remove_answer_leak_from_excerpt(excerpt: str) -> str:
    excerpt = strip_boilerplate(excerpt)
    if not excerpt:
        return ""
    lowered = excerpt.lower()
    cut_positions = []
    for p in ANSWER_LEAK_PATTERNS:
        m = re.search(p, lowered, flags=re.IGNORECASE)
        if m:
            cut_positions.append(m.start())
    if cut_positions:
        excerpt = excerpt[: min(cut_positions)].strip()
    return normalize_space(excerpt)


def choose_benchmark_query(record: dict[str, Any]) -> str:
    title = normalize_space(record.get("title") or "")
    question_text = strip_boilerplate(record.get("question_text") or "")
    if not question_text or contains_pattern(question_text, BOILERPLATE_PATTERNS) or len(question_text) > 250:
        return title
    if title and normalize_key(question_text).startswith(normalize_key(title)):
        return title
    return question_text or title


def build_classification_text(record: dict[str, Any]) -> str:
    title = normalize_space(record.get("title") or "")
    question_text = strip_boilerplate(record.get("question_text") or "")
    excerpt = remove_answer_leak_from_excerpt(record.get("excerpt") or "")
    parts: list[str] = []
    seen: set[str] = set()
    for part in [title, question_text, excerpt]:
        key = normalize_key(part)
        if part and key and key not in seen:
            seen.add(key)
            parts.append(part)
    text = normalize_space(". ".join(parts))
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
    if has_fragment_url(url): flags.append("comment_or_reply_url")
    if contains_pattern(combined, BOILERPLATE_PATTERNS): flags.append("has_boilerplate")
    if contains_pattern(combined, SPAM_PATTERNS): flags.append("possible_spam")
    if len(title) < 15: flags.append("short_title")
    if not title: flags.append("missing_title")
    if not excerpt: flags.append("missing_excerpt")
    if "..." in excerpt or "&..." in excerpt: flags.append("possibly_truncated_excerpt")
    return sorted(set(flags))


def record_type_from_flags(flags: list[str]) -> str:
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
    enriched["record_type"] = record_type_from_flags(enriched["quality_flags"])
    enriched.update(classify_record(enriched))
    return enriched


def dedup_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    def score(r: dict[str, Any]) -> int:
        s = 0
        if r.get("record_type") == "thread": s += 10
        if r.get("excerpt_clean"): s += 3
        if r.get("question_text_clean"): s += 2
        if not r.get("quality_flags"): s += 1
        return s
    for r in records:
        key = r.get("canonical_url") or f"title:{normalize_key(r.get('title_clean') or '')}"
        if key not in best or score(r) > score(best[key]):
            best[key] = r
    return list(best.values())


def write_manual_review_sample(path: Path, records: list[dict[str, Any]], sample_size: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        key = (r.get("legal_domain", "unknown"), r.get("technical_challenge", "unknown"))
        buckets.setdefault(key, []).append(r)
    sampled: list[dict[str, Any]] = []
    per_bucket = max(1, sample_size // max(1, len(buckets)))
    for rows in buckets.values():
        random.shuffle(rows)
        sampled.extend(rows[:per_bucket])
    if len(sampled) < sample_size:
        ids = {id(r) for r in sampled}
        remaining = [r for r in records if id(r) not in ids]
        random.shuffle(remaining)
        sampled.extend(remaining[: sample_size - len(sampled)])
    sampled = sampled[:sample_size]
    cols = ["_raw_line_no", "thread_id", "title_clean", "benchmark_query", "classification_text", "url", "record_type", "quality_flags", "legal_domain", "legal_issue", "technical_challenge", "scope", "review_legal_domain", "review_legal_issue", "review_technical_challenge", "review_scope", "review_note"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sampled:
            row = {c: r.get(c, "") for c in cols}
            row["quality_flags"] = ";".join(r.get("quality_flags") or [])
            w.writerow(row)


def write_summary(path: Path, raw: list[dict[str, Any]], deduped: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counters = {
        "Record type": Counter(r.get("record_type", "unknown") for r in deduped),
        "Scope": Counter(r.get("scope", "unknown") for r in deduped),
        "Legal domain": Counter(r.get("legal_domain", "unknown") for r in deduped),
        "Legal issue": Counter(r.get("legal_issue", "unknown") for r in deduped),
        "Technical challenge": Counter(r.get("technical_challenge", "unknown") for r in deduped),
    }
    flag_counter: Counter[str] = Counter()
    for r in deduped:
        for flag in r.get("quality_flags") or []:
            flag_counter[flag] += 1
    counters["Quality flags"] = flag_counter
    lines = ["# Danluat Hôn nhân & Gia đình - Step 3 V2 Summary", "", f"- Raw records: {len(raw)}", f"- Deduped records: {len(deduped)}", f"- Dropped duplicates: {len(raw) - len(deduped)}", ""]
    for title, counter in counters.items():
        total = sum(counter.values()) or 1
        lines += [f"## {title}", "", "| Label | Count | Ratio |", "|---|---:|---:|"]
        for k, v in counter.most_common():
            lines.append(f"| {k} | {v} | {v/total*100:.1f}% |")
        lines.append("")
    lines += ["## Notes", "", "- V2 ưu tiên title/benchmark_query/excerpt_clean, giảm ảnh hưởng full_question/question_text nhiễu boilerplate.", "- Nhãn vẫn là preliminary labels; cần review tay vòng 2 trước khi dùng làm gold label.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Danluat forum dataset with taxonomy V2.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="data/evaluation/forum_danluat_168_v2")
    ap.add_argument("--sample-size", type=int, default=120)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    raw = read_jsonl(input_path)
    all_labeled = [enrich_record(r) for r in raw]
    deduped = dedup_records(all_labeled)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "danluat_168_v2_all_labeled.jsonl", all_labeled)
    write_jsonl(out_dir / "danluat_168_v2_clean_labeled_dedup.jsonl", deduped)
    write_manual_review_sample(out_dir / "danluat_168_v2_manual_review_sample.csv", deduped, args.sample_size, args.seed)
    write_summary(out_dir / "danluat_168_v2_summary.md", raw, deduped)
    (out_dir / "danluat_168_v2_label_schema.json").write_text(json.dumps(LABEL_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done V2.")
    print(f"Raw records: {len(raw)}")
    print(f"Deduped: {len(deduped)}")
    print(f"Output dir: {out_dir}")

if __name__ == "__main__":
    main()
