# AIBondhu: Bangla Medical Dataset Plan

## Purpose

This document defines a cautious data pipeline for research fine-tuning. It does not certify a model for diagnosis, prescribing, triage, or other clinical use.

## Candidate Bengali QA dataset

- Dataset: [Arittra/bangla_medical_qa_cot_and_non_cot](https://huggingface.co/datasets/Arittra/bangla_medical_qa_cot_and_non_cot)
- Dataset card declares Apache-2.0 and 10,270 rows (6,276 `cot`, 3,994 `non_cot`).
- The card says much of the source material was machine-translated into Bengali and was not fully reviewed by medical professionals. It warns about terminology, numbers, units, synthetic reasoning errors, geographic bias, and conflicting/duplicate answers.
- Therefore this is a **candidate source for manual review**, not ground-truth medical truth.

### Inclusion policy

1. Do not train on the `cot` reasoning traces by default; they may contain generated or translated clinical errors.
2. The `non_cot` subset must be schema-checked, deduplicated, checked for contradictory answers, and placed in a clinician-review queue.
3. The dataset card identifies 1,733 non-CoT examples as MedReason-derived and 2,261 as doctor_qa_bangla-derived. The latter's upstream provenance is not fully verified in the card; keep it quarantined until provenance and rights are confirmed.
4. Do not move any row into the approved training split until a qualified clinician has reviewed the answer and terminology.
5. Keep a source, license, version/revision, reviewer, review date, and decision for every record.

## Evaluation-only source

- [Arittra/BanglaMedBench](https://huggingface.co/datasets/Arittra/BanglaMedBench) describes a 500-question Bengali medical benchmark and explicitly says not to train on it.
- Keep it in a separate evaluation-only split. Do not use its questions or answers for training or tuning.
- Its own card warns that it is machine-translated and small; benchmark scores are not clinical validation.

## Required additional sources before any clinical claim

Prioritize current, publicly licensed and locally relevant guidance from recognized health authorities (for example WHO and applicable national health ministries), with source URL, publication/update date, license/terms, and clinician review recorded. Do not scrape or redistribute copyrighted textbooks, paywalled content, patient records, or datasets with unclear rights.

## Quality gates

- Verify licenses and upstream attribution for every source.
- Remove personal data and any patient-identifying information.
- Detect exact and near duplicates and conflicting answers.
- Validate Bengali medical terminology, drug names, doses, units, age groups, pregnancy guidance, and emergency advice with qualified clinicians.
- Split by source/question family before training to reduce leakage.
- Report performance by specialty, age group, and safety-critical category; include abstention and emergency escalation tests.
- Publish a model card, dataset card, limitations, and evaluation results.

## Training status and compute

No new fine-tuning is started by this document. The repository's existing tiny from-scratch CPU model is not a suitable substitute for fine-tuning a pretrained language model. A GPU runner and a separately reviewed training pipeline are required before meaningful fine-tuning can begin.

## Safety

AIBondhu must not claim to be an MBBS doctor or guarantee 100% accuracy. It must communicate uncertainty, avoid unsupported diagnosis/prescriptions, and direct users to qualified care—especially for emergency symptoms. Passing a public QA benchmark does not establish clinical safety.
