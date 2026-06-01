from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from configs.setting import config
from src.ingestion.pdf_loader import extract_source_text, read_jsonl, resolve_repo_path


def main() -> None:
    dataset_cfg = config.dataset
    raw_dir = resolve_repo_path(getattr(dataset_cfg, "raw_dir", "data/raw/family_law/pdfs"))
    manifest_path = resolve_repo_path(getattr(dataset_cfg, "manifest_path", "data/raw/family_law/manifest.jsonl"))
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty or missing: {manifest_path}")

    print(f"Checking {len(rows)} local source file(s) from manifest: {manifest_path}")
    bad = 0
    for row in rows:
        filename = str(row.get("filename", "")).strip()
        doc_id = str(row.get("doc_id", "")).strip()
        title = str(row.get("title", "")).strip()
        source_path = Path(filename)
        if not source_path.is_absolute():
            direct = ROOT_DIR / source_path
            source_path = direct if direct.exists() else raw_dir / filename
        try:
            text, report = extract_source_text(source_path)
            page_or_para = report.get("page_count") or report.get("paragraph_count") or "-"
            print(
                f"OK  doc_id={doc_id} unit_count={page_or_para} "
                f"chars={len(text)} extractor={report.get('extractor')} file={source_path.name}"
            )
        except Exception as exc:
            bad += 1
            print(f"BAD doc_id={doc_id} title={title} file={source_path}")
            print(f"    {type(exc).__name__}: {exc}")

    if bad:
        raise SystemExit(
            f"Found {bad} invalid/unreadable source file(s). Re-download, OCR, or replace them with DOCX/TXT before ingest."
        )
    print("All local sources look readable.")


if __name__ == "__main__":
    main()
