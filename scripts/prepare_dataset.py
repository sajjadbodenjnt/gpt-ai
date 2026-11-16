import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Validate and deduplicate authorized JSONL text data")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-chars", type=int, default=40)
    args = p.parse_args()
    src, dst = Path(args.input), Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, rejected = set(), 0, 0
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for n, line in enumerate(inp, 1):
            try:
                obj = json.loads(line)
                text = " ".join(str(obj.get("text", "")).split())
                if len(text) < args.min_chars or text in seen:
                    rejected += 1
                    continue
                # Basic guardrail only; this is NOT reliable de-identification.
                lowered = text.lower()
                if any(tag in lowered for tag in ["patient name:", "date of birth:", "mrn:", "social security number:"]):
                    rejected += 1
                    continue
                seen.add(text)
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1
            except (json.JSONDecodeError, AttributeError):
                rejected += 1
    print(f"Kept {kept}; rejected {rejected}; output={dst}")
    print("WARNING: manually verify provenance, license, consent and de-identification before training.")

if __name__ == "__main__":
    main()
