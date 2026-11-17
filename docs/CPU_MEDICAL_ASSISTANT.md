# CPU-first AIBondhu medical evidence assistant

## What this adds

A deterministic, CPU-only evidence retrieval layer. It searches only records marked `review_status=approved` and containing a question, answer, source title, source URL, license, and reviewer role. It returns the source alongside the answer, abstains when evidence is weak, and escalates a small set of emergency phrases.

This is not a neural medical model and does not make the project clinically validated. It is a safer small-device architecture than asking the existing randomly initialized byte-level Transformer to invent medical answers.

## Record format

One JSON object per line. Required fields:
- `question`, `answer`
- `source_title`, `source_url`, `license`
- `review_status` must be `approved`
- `reviewer_role` must identify the qualified reviewer role

Optional: `source_date`, `specialty`, `language`, `reviewer_id` (use a non-identifying internal code, never a person's private details).

The example file is intentionally marked pending and must not be treated as real medical content.

## Validate and query

```bash
python scripts/validate_medical_qa.py --input /path/to/clinician_reviewed.jsonl
python -m aibondhu.medical_retrieval --data /path/to/clinician_reviewed.jsonl --question "জ্বর হলে কী করা উচিত"
```

No network or GPU is needed. The index is built in memory from the local JSONL file.

## Dataset quality rules

Only add material whose license permits the intended use. Record exact source/version/date. A qualified clinician must review the Bengali wording, clinical meaning, units, age/pregnancy caveats, and emergency guidance. Exclude unreviewed machine translations, generated chain-of-thought, identifiable patient data, and records with unclear provenance. Keep evaluation questions separate from retrieval/training records.

## Limits

Character n-gram similarity is a simple lexical matcher; it can miss paraphrases and return a superficially similar but clinically irrelevant question. A similarity score is not a probability or confidence estimate. Review every retrieved answer in context; do not use this tool for diagnosis, prescribing, dosing, or patient-specific decisions. The emergency phrase list is incomplete and cannot reliably detect every emergency. When in doubt, contact local emergency services or a qualified clinician.
