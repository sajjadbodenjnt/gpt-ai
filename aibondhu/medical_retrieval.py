"""CPU-only retrieval over clinician-reviewed medical QA records.

This is an evidence lookup layer, not a diagnostic or treatment system.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

EMERGENCY_TERMS = (
    "বুকে ব্যথা", "তীব্র বুকে ব্যথা", "শ্বাসকষ্ট", "শ্বাস নিতে পারছি না",
    "অজ্ঞান", "খিঁচুনি", "প্রচুর রক্তপাত", "রক্তপাত বন্ধ হচ্ছে না",
    "স্ট্রোক", "মুখ বেঁকে", "হাত অবশ", "বিষ পান", "বিষ খেয়",
    "আত্মহত্যা", "নিজেকে মেরে", "severe chest pain", "difficulty breathing",
    "cannot breathe", "unconscious", "seizure", "heavy bleeding",
    "poisoning", "stroke", "suicidal"
)

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))

def _ngrams(text: str, min_n: int = 2, max_n: int = 4) -> Counter:
    compact = _normalize(text).replace(" ", "")
    padded = f"^{compact}$"
    grams = []
    for n in range(min_n, max_n + 1):
        grams.extend(padded[i:i+n] for i in range(max(0, len(padded)-n+1)))
    return Counter(grams)

def _approved(record: dict) -> bool:
    return (
        record.get("review_status") == "approved"
        and bool(record.get("question", "").strip())
        and bool(record.get("answer", "").strip())
        and bool(record.get("source_url", "").strip())
        and bool(record.get("source_title", "").strip())
        and bool(record.get("license", "").strip())
        and bool(record.get("reviewer_role", "").strip())
    )

class MedicalEvidenceIndex:
    def __init__(self, records: Iterable[dict]):
        self.records = [r for r in records if _approved(r)]
        self.doc_terms = [_ngrams(r["question"]) for r in self.records]
        df = Counter()
        for terms in self.doc_terms:
            df.update(terms.keys())
        total = len(self.doc_terms)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        self.vectors = [self._weighted(t) for t in self.doc_terms]

    @classmethod
    def from_jsonl(cls, path: str | Path):
        records = []
        with Path(path).open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}") from exc
        return cls(records)

    def _weighted(self, counts: Counter) -> dict:
        return {term: (1 + math.log(count)) * self.idf.get(term, 0.0)
                for term, count in counts.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(term, 0.0) for term, value in a.items())
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def answer(self, question: str, threshold: float = 0.22) -> dict:
        normalized = _normalize(question)
        if any(term in normalized for term in EMERGENCY_TERMS):
            return {
                "status": "emergency",
                "message": "এটি জরুরি অবস্থা হতে পারে। এখনই স্থানীয় জরুরি সেবা বা নিকটস্থ হাসপাতালের সঙ্গে যোগাযোগ করুন। AI উত্তরের জন্য অপেক্ষা করবেন না।",
                "evidence": []
            }
        if not self.records:
            return {"status": "insufficient_evidence",
                    "message": "যাচাইকৃত চিকিৎসা-তথ্যভান্ডার এখনো প্রস্তুত নয়। অনুমান করে উত্তর দেওয়া নিরাপদ নয়।",
                    "evidence": []}
        query = self._weighted(_ngrams(question))
        ranked = sorted(
            ((self._cosine(query, vector), i) for i, vector in enumerate(self.vectors)),
            reverse=True
        )
        score, index = ranked[0]
        if score < threshold:
            return {"status": "insufficient_evidence",
                    "message": "যাচাইকৃত তথ্যভান্ডারে এই প্রশ্নের যথেষ্ট মিল পাওয়া যায়নি। একজন যোগ্য চিকিৎসকের পরামর্শ নিন।",
                    "evidence": []}
        record = self.records[index]
        return {
            "status": "retrieved_evidence",
            "message": record["answer"],
            "evidence": [{
                "source_title": record["source_title"],
                "source_url": record["source_url"],
                "source_date": record.get("source_date", ""),
                "similarity": round(score, 4),
                "reviewer_role": record["reviewer_role"]
            }]
        }

def main():
    parser = argparse.ArgumentParser(description="CPU-only AIBondhu evidence lookup")
    parser.add_argument("--data", required=True, help="Clinician-approved QA JSONL")
    parser.add_argument("--question", required=True)
    parser.add_argument("--threshold", type=float, default=0.22)
    args = parser.parse_args()
    index = MedicalEvidenceIndex.from_jsonl(args.data)
    print(json.dumps(index.answer(args.question, args.threshold), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
