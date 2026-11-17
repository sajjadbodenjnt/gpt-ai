# AIBondhu Medical AI

Experimental medical-domain language model trained **from random initialization** (no pretrained weights).

> **Safety:** Research/education prototype only. Not a medical device, diagnostic system, or substitute for a licensed clinician. Never use it for emergencies, diagnosis, treatment selection, medication dosing, or patient-specific decisions.

## Included
- Decoder-only Transformer (causal self-attention, pre-norm, GELU MLP) in PyTorch.
- Reversible UTF-8 byte tokenizer; no pretrained tokenizer download.
- JSONL data validation, deduplication and train/validation split.
- Training, checkpointing, validation loss/perplexity and generation.
- CI smoke tests and a manually triggered GPU workflow.

## Quick start (CPU smoke test)
Python 3.10+ recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --input data/sample.jsonl --output data/processed.jsonl --min-chars 10
python -m aibondhu.train --config configs/smoke.yaml
python -m aibondhu.evaluate --checkpoint outputs/smoke/best.pt --data data/processed.jsonl
```

## Train on an authorized corpus
1. Obtain a corpus you are legally permitted to use for model training. Verify license, source, consent, de-identification, and permitted purposes.
2. Convert it to JSONL, one record per line: `{"text":"educational passage..." }`. Keep private datasets outside Git. Never upload identifiable health data, patient records, names, dates of birth, addresses, or other PHI.
3. Prepare:
```bash
python scripts/prepare_dataset.py --input /path/to/authorized.jsonl --output data/processed.jsonl
```
4. Edit `configs/train.yaml` for available compute and dataset size, then run:
```bash
python -m aibondhu.train --config configs/train.yaml
```

The included sample is synthetic and intended only to test the pipeline. It is far too small to produce a useful medical language model. This is a small from-scratch baseline, not a competitive clinical LLM. Meaningful language capability requires a large, high-quality, legally usable corpus and substantial compute.

## GitHub Actions
- **CI** runs tests and a tiny CPU training smoke test on pushes and pull requests.
- **GPU training** is manual: Actions → Train AIBondhu (GPU) → Run workflow. It requires a trusted self-hosted runner labelled `self-hosted, linux, x64, gpu`, with a supported Python/PyTorch/CUDA setup and an authorized JSONL dataset available at the selected runner path. GitHub-hosted runners do not guarantee GPU access.
- Outputs are uploaded as workflow artifacts; trained weights are not committed to Git.

## Dataset schema
Each line: `{"text":"..." }`. The preparation script removes empty/duplicate/short lines and checks a few obvious identifier labels. This is not reliable de-identification or legal review. Maintain a dataset manifest recording source, retrieval date, version, license, intended use, and filtering steps.

## Evaluation and intended use
The evaluation script reports language-model validation loss/perplexity only. It does not establish clinical accuracy, safety, fairness, or readiness. Before any real-world health use, require independent expert-led validation, privacy/security review, bias and robustness testing, red-team evaluation, regulatory review, and human oversight. Do not use outputs for diagnosis, triage, prescribing, dosage, or patient-specific care.

## License
Code: MIT (see LICENSE). Dataset and model artifacts retain all obligations of their source licenses; this repository does not grant rights to third-party data.


## AIBondhu from-scratch CPU experiment

A separate experimental pipeline builds a small GPT and ByteLevel BPE tokenizer from random initialization (no company pretrained weights/tokenizer): see [Bengali instructions](docs/SCRATCH_TRAINING_BN.md) and `.github/workflows/train-scratch-5h.yml`. The workflow targets up to 300 minutes of CPU training on licensed PMC Open Access biomedical articles. Five hours of CPU training cannot create an extremely intelligent or clinically reliable model; the output is a research prototype only. Checkpoint and tokenizer are uploaded as an Actions artifact, not committed to Git.
