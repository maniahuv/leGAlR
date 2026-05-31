from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def main():
    manifest_path = ROOT_DIR / "data" / "evaluation" / "indexed_manifest.json"
    testset_path = ROOT_DIR / "data" / "evaluation" / "legal_test_cases.json"

    if not manifest_path.exists():
        print("Missing indexed_manifest.json. Run python scripts/ingest.py first.")
        return
    if not testset_path.exists():
        print("Missing legal_test_cases.json. Run python scripts/generate_test_cases.py first.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_cases = json.loads(testset_path.read_text(encoding="utf-8"))

    indexed_ids = {str(x).strip() for x in manifest.get("indexed_doc_ids", [])}
    gold_ids = set()
    for case in test_cases:
        gold_ids.update(str(x).strip() for x in case.get("relevant_ids", []) if str(x).strip())

    missing = gold_ids - indexed_ids
    print(f"Indexed docs: {len(indexed_ids)}")
    print(f"Gold docs:    {len(gold_ids)}")
    print(f"Missing gold: {len(missing)}")
    if missing:
        print("Sample missing:", list(sorted(missing))[:20])
    else:
        print("OK: every gold doc_id exists in the indexed corpus.")


if __name__ == "__main__":
    main()
