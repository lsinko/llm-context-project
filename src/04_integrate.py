import json
import os
import re
import time
from typing import Dict, Any, Optional

import pandas as pd
import requests

KAGGLE_CLEAN = "data/processed/kaggle_clean.csv"
CANDIDATES_JSON = "data/raw/hf_candidates_by_row.json"

OUT_MAP = "data/processed/kaggle_to_hf_map.csv"
OUT_MERGED = "data/processed/merged_llm_data.csv"
OUT_REPO_LEVEL = "data/processed/merged_llm_data_repo_level.csv"

HF_CACHE = "data/raw/hf_metrics_by_repo.json"
HF_MODEL_URL = "https://huggingface.co/api/models/{}"

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


def family_token(model_name: str, provider: str) -> str:
    m = str(model_name).strip().lower()
    mm = re.fullmatch(r"(llama)-(\d+)", m)
    if mm:
        return f"{mm.group(1)}-{mm.group(2)}"
    return ""


def choose_best_candidate(row: pd.Series, candidates: list[dict]) -> Optional[str]:
    provider_norm = str(row["provider"]).strip().lower()
    if provider_norm in CLOSED:
        return None

    model_name = str(row["model_name"])
    prefixes = provider_prefixes(provider_norm)
    tok = family_token(model_name, provider_norm)

    filt = []
    for c in candidates:
        rid = (c.get("id") or "").strip()
        if not rid:
            continue
        if prefixes and not any(rid.startswith(pref) for pref in prefixes):
            continue
        if tok and tok not in rid.lower():
            continue
        filt.append(c)

    if not filt:
        return None

    def key(c):
        d_all = c.get("downloadsAllTime")
        d = c.get("downloads")
        l = c.get("likes")
        return (
            0 if d is None else int(d),
            0 if l is None else int(l),
            0 if d_all is None else int(d_all),
        )

    filt.sort(key=key, reverse=True)
    return (filt[0].get("id") or "").strip() or None


def load_cache(path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_hf_metrics(repo_id: str) -> Dict[str, Any]:
    url = HF_MODEL_URL.format(repo_id)
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return {
            "hf_repo_id": repo_id,
            "hf_status": f"http_{r.status_code}",
            "hf_likes": None,
            "hf_downloads": None,
            "hf_downloads_all_time": None,
        }
    obj = r.json()
    return {
        "hf_repo_id": repo_id,
        "hf_status": "ok",
        "hf_likes": obj.get("likes"),
        "hf_downloads": obj.get("downloads"),
        "hf_downloads_all_time": obj.get("downloadsAllTime"),
    }


def main():
    df = pd.read_csv(KAGGLE_CLEAN)

    with open(CANDIDATES_JSON, "r", encoding="utf-8") as f:
        cand_by_row = json.load(f)

    map_rows = []
    for _, row in df.iterrows():
        row_id = int(row["kaggle_row_id"])
        rec = cand_by_row.get(str(row_id))
        candidates = (rec or {}).get("candidates", [])
        best = choose_best_candidate(row, candidates)

        map_rows.append(
            {
                "kaggle_row_id": row_id,
                "model_name": row["model_name"],
                "provider": row["provider"],
                "hf_repo_id": best or "",
            }
        )

    mp = pd.DataFrame(map_rows)
    os.makedirs("data/processed", exist_ok=True)
    mp.to_csv(OUT_MAP, index=False, encoding="utf-8-sig")

    merged = df.merge(mp[["kaggle_row_id", "hf_repo_id"]], on="kaggle_row_id", how="left")
    merged["hf_repo_id"] = merged["hf_repo_id"].astype(str).str.strip()
    merged = merged[merged["hf_repo_id"] != ""].copy()

    cache = load_cache(HF_CACHE)
    repo_ids = sorted(merged["hf_repo_id"].unique().tolist())

    for rid in repo_ids:
        if rid in cache:
            continue
        cache[rid] = fetch_hf_metrics(rid)
        save_cache(HF_CACHE, cache)
        time.sleep(0.2)

    metrics_df = pd.DataFrame(list(cache.values()))
    merged = merged.merge(metrics_df, on="hf_repo_id", how="left")

    merged = merged[(merged["hf_status"] == "ok") & (~merged["hf_downloads"].isna())].copy()
    merged.to_csv(OUT_MERGED, index=False, encoding="utf-8-sig")

    numeric_cols = [
        "context_window", "speed_tokens_per_sec", "latency_sec",
        "benchmark_mmlu", "benchmark_chatbot_arena",
        "price_per_million_tokens", "training_dataset_size",
        "compute_power", "energy_efficiency",
        "quality_rating", "speed_rating", "price_rating",
        "open_source",
    ]
    numeric_cols = [c for c in numeric_cols if c in merged.columns]

    repo_level = (
        merged
        .groupby("hf_repo_id", as_index=False)
        .agg(
            provider=("provider", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            n_kaggle_rows=("kaggle_row_id", "count"),
            context_window=("context_window", "max"),
            speed_tokens_per_sec=("speed_tokens_per_sec", "median"),
            latency_sec=("latency_sec", "median"),
            benchmark_mmlu=("benchmark_mmlu", "median"),
            benchmark_chatbot_arena=("benchmark_chatbot_arena", "median"),
            open_source=("open_source", "max"),
            price_per_million_tokens=("price_per_million_tokens", "median"),
            training_dataset_size=("training_dataset_size", "median"),
            compute_power=("compute_power", "median"),
            energy_efficiency=("energy_efficiency", "median"),
            quality_rating=("quality_rating", "median"),
            speed_rating=("speed_rating", "median"),
            price_rating=("price_rating", "median"),
            hf_status=("hf_status", "first"),
            hf_likes=("hf_likes", "max"),
            hf_downloads=("hf_downloads", "max"),
            hf_downloads_all_time=("hf_downloads_all_time", "max"),
        )
    )

    repo_level.to_csv(OUT_REPO_LEVEL, index=False, encoding="utf-8-sig")

    print("Saved map:", OUT_MAP, "rows=", len(mp))
    print("Saved merged (row-level):", OUT_MERGED, "rows=", len(merged))
    print("Saved merged (repo-level):", OUT_REPO_LEVEL, "rows=", len(repo_level))


if __name__ == "__main__":
    main()
