from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABELS = ["legal_domain", "legal_issue", "technical_challenge", "scope"]


def nonempty(v) -> bool:
    return v is not None and str(v).strip() != "" and str(v).strip().lower() != "nan"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute manual-review correction statistics.")
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--out", default="data/evaluation/forum_danluat_168_v2/manual_review_stats.md")
    args = ap.parse_args()

    path = Path(args.review_csv)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    lines = ["# Manual Review Statistics", "", f"- Rows: {len(rows)}", ""]
    lines += ["| Label | Corrections | Estimated accuracy |", "|---|---:|---:|"]
    any_corrected = 0
    for label in LABELS:
        col = f"review_{label}"
        corrections = sum(1 for r in rows if nonempty(r.get(col)))
        acc = (len(rows) - corrections) / max(1, len(rows)) * 100
        lines.append(f"| `{label}` | {corrections}/{len(rows)} | {acc:.1f}% |")
    for r in rows:
        if any(nonempty(r.get(f"review_{label}")) for label in LABELS):
            any_corrected += 1
    lines += ["", f"- Rows with at least one correction: {any_corrected}/{len(rows)}", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
