"""Stream the official PMC Open Access dataset into AIBondhu JSONL.

Only CC0, CC BY and CC BY-SA records are admitted by default. This deliberately
excludes non-commercial, no-derivatives, unknown/custom-license records and
retracted articles. The upstream dataset's per-record license remains the
controlling term; retain the metadata manifest for attribution.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

ALLOWED = {"CC0", "CC BY", "CC BY-SA"}

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/pmc_open_access.jsonl")
    p.add_argument("--manifest", default="data/pmc_manifest.jsonl")
    p.add_argument("--max-articles", type=int, default=10000)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--split", default="train")
    args = p.parse_args()

    # This dataset currently uses a repository-provided loading script.
    # datasets 4+ removed script-based loading; requirements pins v3.
    stream = load_dataset(
        "pmc/open_access",
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )
    out_path, manifest_path = Path(args.output), Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    kept = rejected = 0
    with out_path.open("w", encoding="utf-8") as out, manifest_path.open("w", encoding="utf-8") as manifest:
        for row in stream:
            license_name = str(row.get("license", "")).strip().upper()
            retracted = str(row.get("retracted", "")).strip().lower()
            text = clean(row.get("text", ""))
            if license_name not in ALLOWED or retracted in {"yes", "true", "1"} or len(text) < args.min_chars:
                rejected += 1
                continue
            text = text[:args.max_chars]
            record = {
                "text": text,
                "pmid": str(row.get("pmid", "")),
                "accession_id": str(row.get("accession_id", "")),
                "license": license_name,
                "citation": str(row.get("citation", "")),
            }
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\\n")
            manifest.write(json.dumps({k: v for k, v in record.items() if k != "text"}, ensure_ascii=False) + "\\n")
            kept += 1
            if kept >= args.max_articles:
                break
    if kept < 100:
        raise SystemExit(f"Only {kept} eligible records found; refusing to start training.")
    print(f"Eligible PMC articles: {kept}; skipped: {rejected}")
    print(f"Training text: {out_path}; attribution/license manifest: {manifest_path}")
    print("Source: https://huggingface.co/datasets/pmc/open_access")
    print("Training is research/education only, not clinical validation.")

if __name__ == "__main__":
    main()
