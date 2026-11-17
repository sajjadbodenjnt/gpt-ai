"""Download a license-filtered PMC sample via the official PMC Cloud Service.

The legacy PMC FTP bulk directories were retired in August 2026. This uses
NCBI E-utilities plus the supported PMC S3 Cloud Service. Only CC0, CC BY,
and CC BY-SA are admitted; retracted and other-license records are excluded.
"""
import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests

BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ALLOWED = {"CC0", "CC BY", "CC BY-SA"}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AIBondhuResearch/0.1 (educational research)"})

def get(url, **kwargs):
    response = SESSION.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/pmc_open_access.jsonl")
    p.add_argument("--manifest", default="data/pmc_manifest.jsonl")
    p.add_argument("--max-articles", type=int, default=100)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--min-chars", type=int, default=500)
    args = p.parse_args()

    query = '((cc0_license[Filter] OR cc_by_license[Filter] OR cc_by-sa_license[Filter]) AND (open_access[Filter] OR author_manuscript[Filter]))'
    search = get(ESEARCH, params={
        "db": "pmc", "term": query,
        "retmax": max(args.max_articles * 5, 500), "retmode": "json"
    })
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise SystemExit("NCBI E-utilities returned no eligible PMC IDs; refusing to train.")

    out_path, manifest_path = Path(args.output), Path(args.manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = errors = 0

    with out_path.open("w", encoding="utf-8") as out, manifest_path.open("w", encoding="utf-8") as manifest:
        for pmcid_num in ids:
            if kept >= args.max_articles:
                break
            pmcid = "PMC" + str(pmcid_num)
            try:
                listing = get(BASE, params={
                    "list-type": "2", "delimiter": "/", "prefix": pmcid + "."
                })
                root = ET.fromstring(listing.content)
                prefixes = sorted({
                    node.text for node in root.findall(".//{*}CommonPrefixes/{*}Prefix")
                    if node.text
                })
                for prefix in prefixes:
                    version_name = prefix.rstrip("/").split("/")[-1]
                    meta_url = BASE + "metadata/" + quote(version_name + ".json")
                    meta = get(meta_url).json()
                    license_name = str(meta.get("license_code", "")).strip().upper()
                    if license_name not in ALLOWED or str(meta.get("is_retracted", "")).lower() in {"yes", "true", "1"}:
                        rejected += 1
                        continue
                    text_url = meta.get("text_url") or (BASE + quote(prefix + version_name + ".txt", safe="/"))
                    # PMC metadata may publish an s3:// URI; translate it to the public HTTPS object endpoint.
                    if text_url.startswith("s3://pmc-oa-opendata/"):
                        text_url = "https://pmc-oa-opendata.s3.amazonaws.com/" + quote(text_url[len("s3://pmc-oa-opendata/"):], safe="/?=&")
                    article_text = clean(get(text_url).text)
                    if len(article_text) < args.min_chars:
                        rejected += 1
                        continue
                    article_text = article_text[:args.max_chars]
                    out.write(json.dumps({"text": article_text}, ensure_ascii=False) + "\n")
                    manifest.write(json.dumps({
                        "pmcid": pmcid,
                        "version": str(meta.get("version", "")),
                        "pmid": str(meta.get("pmid", "")),
                        "doi": str(meta.get("doi", "")),
                        "title": str(meta.get("title", "")),
                        "citation": str(meta.get("citation", "")),
                        "license": license_name,
                        "source": text_url,
                        "is_retracted": False,
                    }, ensure_ascii=False) + "\n")
                    kept += 1
                    if kept >= args.max_articles:
                        break
                time.sleep(0.12)
            except (requests.RequestException, ValueError, ET.ParseError, KeyError) as exc:
                errors += 1
                print(f"Skipping {pmcid}: {type(exc).__name__}: {exc}")

    if kept < 100:
        raise SystemExit(
            f"Only {kept} eligible articles collected (rejected={rejected}, errors={errors}); refusing to start training."
        )
    print(f"Eligible PMC articles: {kept}; rejected: {rejected}; retrieval errors: {errors}")
    print(f"Training text: {out_path}; attribution/license manifest: {manifest_path}")
    print("Source: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/")
    print("Research/education only; this model is not clinically validated.")

if __name__ == "__main__":
    main()
