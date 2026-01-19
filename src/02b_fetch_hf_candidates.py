import json
import os
import time
import pandas as pd
import requests

KAGGLE_CLEAN = "data/processed/kaggle_clean.csv"
OUT_JSON = "data/raw/hf_candidates_by_row.json"

HF_SEARCH_URL = "https://huggingface.co/api/models"

CLOSED = {"openai", "google", "anthropic", "aws"}


def provider_prefixes(provider: str):
    p = str(provider).strip().lower()
    if "meta" in p:
        return ["meta-llama/"]
    if "deepseek" in p:
        return ["deepseek-ai/"]
    if "mistral" in p:
        return ["mistralai/"]
    if "cohere" in p:
        return ["CohereLabs/", "CohereForAI/", "Cohere/"]
    return []


def hf_search(query: str, limit: int = 50):
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
        "expand[]": ["downloads", "downloadsAllTime", "likes"],
    }
    r = requests.get(HF_SEARCH_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


import re

def family_token(model_name: str, provider: str) -> str:
    m = str(model_name).strip().lower()
    p = str(provider).strip().lower()

    mm = re.fullmatch(r"(llama)-(\d+)", m)
    if mm:
        return f"{mm.group(1)}-{mm.group(2)}"

    return ""

def build_query(model_name: str, provider: str) -> str:
    tok = family_token(model_name, provider)
    if tok:
        return tok
    return f"{str(model_name).strip()} {str(provider).strip()}".strip()



def main():
    df = pd.read_csv(KAGGLE_CLEAN)
    os.makedirs("data/raw", exist_ok=True)

    out = {}

    for _, row in df.iterrows():
        row_id = int(row["kaggle_row_id"])
        model_name = str(row["model_name"])
        provider = str(row["provider"])

        if provider.strip().lower() in CLOSED:
            out[str(row_id)] = {
                "kaggle_row_id": row_id,
                "model_name": model_name,
                "provider": provider,
                "query": None,
                "candidates": [],
                "note": "closed_source_provider_skipped",
            }
            continue

        q = build_query(model_name, provider)
        items = hf_search(q, limit=50)

        prefixes = provider_prefixes(provider)
        if prefixes:
            items = [
                it for it in items
                if any((it.get("id") or "").startswith(pref) for pref in prefixes)
            ]

        candidates = []
        for it in items[:50]:
            candidates.append(
                {
                    "id": it.get("id"),
                    "likes": it.get("likes"),
                    "downloads": it.get("downloads"),
                    "downloadsAllTime": it.get("downloadsAllTime"),
                }
            )

        out[str(row_id)] = {
            "kaggle_row_id": row_id,
            "model_name": model_name,
            "provider": provider,
            "query": q,
            "candidates": candidates,
        }

        time.sleep(0.15)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", OUT_JSON, "rows=", len(out))


if __name__ == "__main__":
    main()
