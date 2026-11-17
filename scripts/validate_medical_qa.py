"""Validate the strict schema used by AIBondhu's evidence retrieval layer."""
import argparse
import json
from pathlib import Path

REQUIRED = ("question", "answer", "source_title", "source_url", "license",
            "review_status", "reviewer_role")

def validate(path: Path):
    errors = []
    count = 0
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"line {line_no}: invalid JSON")
                continue
            missing = [key for key in REQUIRED if not str(row.get(key, "")).strip()]
            if missing:
                errors.append(f"line {line_no}: missing {', '.join(missing)}")
                continue
            if row["review_status"] != "approved":
                errors.append(f"line {line_no}: review_status must be approved")
            if not row["source_url"].startswith(("https://", "http://")):
                errors.append(f"line {line_no}: source_url must be HTTP(S)")
            key = " ".join(row["question"].lower().split())
            if key in seen:
                errors.append(f"line {line_no}: duplicate question")
            seen.add(key)
    return count, errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    count, errors = validate(Path(args.input))
    print(f"Records: {count}; errors: {len(errors)}")
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
