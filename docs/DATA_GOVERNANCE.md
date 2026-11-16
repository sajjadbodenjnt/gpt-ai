# Data governance checklist

Before adding any corpus:
- Confirm the source, copyright/license, terms of use, and redistribution/training permissions.
- Confirm consent and lawful basis where required; remove direct and indirect identifiers.
- Never upload patient charts, medical records, private messages, credentials, or protected health information.
- Keep a dataset manifest: source URL, retrieval date, version, license, intended use, filtering steps, and known limitations.
- Keep private datasets out of Git; use controlled storage and least-privilege access.
- Split by source/patient where appropriate to reduce leakage; do not put the same passage in train and validation.
- Obtain domain-expert review and document demographic/clinical coverage and limitations.

The included script performs only basic formatting/deduplication and a few keyword checks. It is not a de-identification or legal-compliance tool.
