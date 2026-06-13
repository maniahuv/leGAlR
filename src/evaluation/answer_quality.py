from __future__ import annotations

"""Utilities for manual answer-quality evaluation of the Legal RAG system.

This module is deliberately lightweight: it does not judge the legal answer by
itself. It prepares a reviewable CSV/JSONL so a human reviewer can score
correctness, groundedness, citation accuracy, completeness, clarity, and safety.
"""

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FULL_DOC_MARKERS = {"", "*", "__full_doc__", None}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    title: str = ""
    source: str = ""
    url: str = ""
    legal_domain: str = ""
    legal_issue: str = ""
    technical_challenge: str = ""
    scope: str = ""
    difficulty: str = ""
    scenario: str = ""
    relevant_ids: tuple[str, ...] = ()
    relevant_doc_ids_full: tuple[str, ...] = ()
    relevant_articles: tuple[str, ...] = ()
    required_pairs: tuple[tuple[str, str], ...] = ()
    raw: dict[str, Any] | None = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Try JSON first, then comma/semicolon split.
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x.strip() for x in re.split(r"[;,]", value) if x.strip()]
    return [value]


def _short_doc_id(doc_id: str) -> str:
    """Normalize long corpus document IDs to the benchmark short form.

    Examples:
    - 52_2014_QH13_luat_hon_nhan_gia_dinh -> 52_2014_QH13
    - 123_2015_ND_CP_huong_dan_luat_ho_tich -> 123_2015_ND_CP
    - 54_2022_AL_quyen_nuoi_con_duoi_36_thang -> 54_2022_AL
    """
    text = str(doc_id or "").strip()
    if not text:
        return ""
    m = re.match(r"^(\d+_\d{4}_(?:QH\d+|ND_CP|TT_BTP|AL|TTLT_[A-Z_]+))", text)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+_\d{4}_[A-Z0-9_]+)", text)
    if m:
        return m.group(1)
    return text


def _article_number(article: Any) -> str:
    text = str(article or "").strip()
    if not text or text in FULL_DOC_MARKERS:
        return "__full_doc__"
    m = re.search(r"(?:Điều\s*)?(\d+[a-zA-Z]?)", text, flags=re.IGNORECASE)
    return m.group(1) if m else text


def parse_required_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in _as_list(value):
        doc_id = ""
        article = "__full_doc__"
        if isinstance(item, dict):
            doc_id = str(
                item.get("doc_id")
                or item.get("doc_id_short")
                or item.get("relevant_id")
                or item.get("doc_id_full")
                or ""
            ).strip()
            article = _article_number(
                item.get("article_number")
                or item.get("article")
                or item.get("article_id")
                or item.get("dieu")
                or ""
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            doc_id = str(item[0]).strip()
            article = _article_number(item[1])
        elif isinstance(item, str):
            # Accept forms such as "52_2014_QH13:81" or "52_2014_QH13 Điều 81".
            text = item.strip()
            if not text:
                continue
            if ":" in text:
                left, right = text.split(":", 1)
                doc_id = left.strip()
                article = _article_number(right)
            else:
                m = re.search(r"(.+?)\s+(?:Điều\s*)?(\d+[a-zA-Z]?)$", text, flags=re.IGNORECASE)
                if m:
                    doc_id = m.group(1).strip()
                    article = _article_number(m.group(2))
                else:
                    doc_id = text
                    article = "__full_doc__"
        if doc_id:
            pair = (_short_doc_id(doc_id), article)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        rows = loaded.get("cases", loaded) if isinstance(loaded, dict) else loaded
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported benchmark format: {path.suffix}")

    cases: list[BenchmarkCase] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        query = str(row.get("query") or row.get("question") or row.get("title") or "").strip()
        if not query:
            continue
        relevant_ids = tuple(_short_doc_id(str(x)) for x in _as_list(row.get("relevant_ids") or row.get("expected_docs")) if str(x).strip())
        relevant_full = tuple(str(x).strip() for x in _as_list(row.get("relevant_doc_ids_full")) if str(x).strip())
        relevant_articles = tuple(_article_number(x) for x in _as_list(row.get("relevant_articles") or row.get("expected_articles")) if str(x).strip())
        required_pairs = parse_required_pairs(row.get("required_pairs"))

        cases.append(
            BenchmarkCase(
                id=str(row.get("id") or row.get("case_id") or f"case_{idx:04d}"),
                query=query,
                title=str(row.get("title") or ""),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or ""),
                legal_domain=str(row.get("legal_domain") or ""),
                legal_issue=str(row.get("legal_issue") or ""),
                technical_challenge=str(row.get("technical_challenge") or ""),
                scope=str(row.get("scope") or ""),
                difficulty=str(row.get("difficulty") or ""),
                scenario=str(row.get("scenario") or ""),
                relevant_ids=relevant_ids,
                relevant_doc_ids_full=relevant_full,
                relevant_articles=relevant_articles,
                required_pairs=required_pairs,
                raw=row,
            )
        )
    return cases


def sample_cases(
    cases: list[BenchmarkCase],
    limit: int | None = None,
    seed: int = 42,
    mode: str = "stratified",
) -> list[BenchmarkCase]:
    if limit is None or limit <= 0 or limit >= len(cases):
        return list(cases)

    rng = random.Random(seed)
    if mode == "first":
        return list(cases[:limit])
    if mode == "random":
        return rng.sample(cases, limit)

    # Stratify by legal_domain + technical_challenge, then fill any remainder.
    buckets: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        key = f"{case.legal_domain or 'unknown'}::{case.technical_challenge or 'unknown'}"
        buckets.setdefault(key, []).append(case)

    selected: list[BenchmarkCase] = []
    bucket_items = list(buckets.items())
    rng.shuffle(bucket_items)

    # At least one from as many buckets as possible.
    for _, bucket in bucket_items:
        if len(selected) >= limit:
            break
        selected.append(rng.choice(bucket))

    if len(selected) < limit:
        selected_ids = {case.id for case in selected}
        remaining = [case for case in cases if case.id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: limit - len(selected)])

    # Keep deterministic output order close to benchmark order.
    order = {case.id: i for i, case in enumerate(cases)}
    selected.sort(key=lambda c: order.get(c.id, 10**9))
    return selected


def doc_to_review_dict(doc: Any, rank: int) -> dict[str, Any]:
    meta = getattr(doc, "metadata", None) or {}
    content = getattr(doc, "page_content", "") or ""
    return {
        "rank": rank,
        "doc_id": str(meta.get("doc_id") or ""),
        "doc_id_short": _short_doc_id(str(meta.get("doc_id") or "")),
        "title": str(meta.get("title") or meta.get("doc_title") or ""),
        "article": str(meta.get("article") or ""),
        "clause": str(meta.get("clause") or ""),
        "route": str(meta.get("route") or ""),
        "status": str(meta.get("tinh_trang_hieu_luc") or ""),
        "snippet": " ".join(content.split())[:800],
    }


def retrieved_pairs_from_docs(docs: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        d_id = _short_doc_id(str(meta.get("doc_id") or ""))
        article = _article_number(meta.get("article"))
        if d_id:
            pair = (d_id, article)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def compute_retrieval_flags(case: BenchmarkCase, docs: Iterable[Any]) -> dict[str, Any]:
    retrieved_pairs = retrieved_pairs_from_docs(docs)
    retrieved_doc_ids = {p[0] for p in retrieved_pairs}
    expected_doc_ids = {_short_doc_id(x) for x in case.relevant_ids} | {_short_doc_id(x) for x in case.relevant_doc_ids_full}
    required = set(case.required_pairs)

    doc_hit = bool(expected_doc_ids & retrieved_doc_ids) if expected_doc_ids else False
    pair_hits = sorted(required & set(retrieved_pairs))
    req_pair_recall = len(pair_hits) / len(required) if required else ""

    expected_articles = {a for a in case.relevant_articles if a and a != "__full_doc__"}
    retrieved_articles = {p[1] for p in retrieved_pairs if p[1] and p[1] != "__full_doc__"}
    article_hit = bool(expected_articles & retrieved_articles) if expected_articles else False

    return {
        "retrieval_doc_hit": doc_hit,
        "retrieval_article_hit": article_hit,
        "retrieval_pair_hits": pair_hits,
        "retrieval_req_pair_recall": req_pair_recall,
        "retrieved_pairs": retrieved_pairs,
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
    }


def expected_pairs_to_text(case: BenchmarkCase) -> str:
    if case.required_pairs:
        return "; ".join(f"{d}:{a}" for d, a in case.required_pairs)
    docs = ", ".join(case.relevant_ids or case.relevant_doc_ids_full)
    arts = ", ".join(case.relevant_articles)
    return f"docs={docs}; articles={arts}".strip("; ")


def make_review_row(
    *,
    case: BenchmarkCase,
    docs: list[Any],
    answer: str,
    timings: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    flags = compute_retrieval_flags(case, docs)
    doc_dicts = [doc_to_review_dict(doc, rank=i) for i, doc in enumerate(docs, start=1)]

    return {
        "id": case.id,
        "query": case.query,
        "title": case.title,
        "legal_domain": case.legal_domain,
        "legal_issue": case.legal_issue,
        "technical_challenge": case.technical_challenge,
        "scope": case.scope,
        "difficulty": case.difficulty,
        "scenario": case.scenario,
        "expected_docs": json.dumps(list(case.relevant_ids or case.relevant_doc_ids_full), ensure_ascii=False),
        "expected_articles": json.dumps(list(case.relevant_articles), ensure_ascii=False),
        "expected_pairs": expected_pairs_to_text(case),
        "retrieved_doc_ids": json.dumps(flags["retrieved_doc_ids"], ensure_ascii=False),
        "retrieved_pairs": json.dumps([f"{d}:{a}" for d, a in flags["retrieved_pairs"]], ensure_ascii=False),
        "retrieved_top5": json.dumps(doc_dicts, ensure_ascii=False),
        "retrieval_doc_hit": flags["retrieval_doc_hit"],
        "retrieval_article_hit": flags["retrieval_article_hit"],
        "retrieval_req_pair_recall": flags["retrieval_req_pair_recall"],
        "answer": answer,
        "answer_chars": len(answer or ""),
        "mentions_insufficient_context": any(x in (answer or "").lower() for x in ["chưa đủ", "không đủ", "thiếu căn cứ", "ngữ cảnh chưa đủ"]),
        "error": error,
        "timings": json.dumps(timings or {}, ensure_ascii=False),
        # Human review columns. Fill manually: 0=sai, 1=đạt một phần, 2=đạt tốt.
        "correctness_score": "",
        "groundedness_score": "",
        "citation_score": "",
        "completeness_score": "",
        "clarity_score": "",
        "safety_score": "",
        "final_label": "",
        "review_note": "",
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


RUBRIC_MD = """# Answer Quality Evaluation Rubric

Chấm thủ công từng câu trả lời theo thang 0–2.

## 1. Correctness — Đúng pháp luật
- 0: Kết luận sai hoặc trái căn cứ pháp luật.
- 1: Kết luận gần đúng nhưng thiếu điều kiện/ngoại lệ quan trọng.
- 2: Kết luận đúng, có nêu điều kiện/ngoại lệ cần thiết.

## 2. Groundedness — Bám ngữ cảnh truy xuất
- 0: Trả lời dựa vào suy đoán hoặc kiến thức ngoài context.
- 1: Có bám context nhưng còn thêm ý không được hỗ trợ rõ.
- 2: Các ý chính đều được hỗ trợ bởi tài liệu truy xuất.

## 3. Citation accuracy — Viện dẫn chính xác
- 0: Viện dẫn sai văn bản/điều luật hoặc bịa căn cứ.
- 1: Viện dẫn đúng một phần nhưng thiếu căn cứ chính hoặc lẫn căn cứ phụ.
- 2: Viện dẫn đúng văn bản và điều/khoản quan trọng.

## 4. Completeness — Đủ ý
- 0: Bỏ sót phần chính của câu hỏi.
- 1: Trả lời được ý chính nhưng thiếu thủ tục/điều kiện/hệ quả quan trọng.
- 2: Trả lời đủ kết luận, căn cứ, áp dụng và lưu ý cần thiết.

## 5. Clarity — Dễ hiểu với người dùng phổ thông
- 0: Rối, lặp câu hỏi, hoặc quá khó hiểu.
- 1: Hiểu được nhưng còn dài dòng/thiếu cấu trúc.
- 2: Rõ ràng, trực tiếp, có cấu trúc tốt.

## 6. Safety — An toàn pháp lý
- 0: Cam kết chắc chắn kết quả xử lý/xét xử hoặc bỏ qua thiếu dữ kiện.
- 1: Có cảnh báo nhưng chưa rõ.
- 2: Nêu đúng giới hạn, điều kiện, dữ kiện cần bổ sung và không cam kết quá mức.

## Gợi ý final_label
- pass: đa số tiêu chí đạt 2, không có lỗi correctness/citation nghiêm trọng.
- partial: có ích nhưng cần sửa một vài điểm.
- fail: sai luật, bịa căn cứ, hoặc không trả lời được câu hỏi.
"""


def write_rubric(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RUBRIC_MD, encoding="utf-8")
