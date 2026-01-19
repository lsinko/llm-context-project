import os
import sqlite3
import pandas as pd

ROW_LEVEL_CSV = "data/processed/merged_llm_data.csv"
REPO_LEVEL_CSV = "data/processed/merged_llm_data_repo_level.csv"
DB_PATH = "data/processed/llm_context.db"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    if not os.path.exists(ROW_LEVEL_CSV):
        raise FileNotFoundError(f"Nedostaje {ROW_LEVEL_CSV}. Prvo pokreni 04_integrate.py.")
    if not os.path.exists(REPO_LEVEL_CSV):
        raise FileNotFoundError(f"Nedostaje {REPO_LEVEL_CSV}. Prvo pokreni 04_integrate.py.")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    row_df = pd.read_csv(ROW_LEVEL_CSV)
    repo_df = pd.read_csv(REPO_LEVEL_CSV)

    row_df = _clean_columns(row_df)
    repo_df = _clean_columns(repo_df)

    row_numeric = [
        "kaggle_row_id",
        "context_window",
        "speed_tokens_per_sec",
        "latency_sec",
        "benchmark_mmlu",
        "benchmark_chatbot_arena",
        "open_source",
        "price_per_million_tokens",
        "training_dataset_size",
        "compute_power",
        "energy_efficiency",
        "quality_rating",
        "speed_rating",
        "price_rating",
        "hf_likes",
        "hf_downloads",
        "hf_downloads_all_time",
    ]
    repo_numeric = [
        "n_kaggle_rows",
        "context_window",
        "speed_tokens_per_sec",
        "latency_sec",
        "benchmark_mmlu",
        "benchmark_chatbot_arena",
        "hf_likes",
        "hf_downloads",
        "hf_downloads_all_time",
    ]

    row_df = _coerce_numeric(row_df, row_numeric)
    repo_df = _coerce_numeric(repo_df, repo_numeric)

    if "kaggle_row_id" not in row_df.columns:
        raise ValueError("U row-level CSV-u nedostaje stupac kaggle_row_id. Provjeri 03_clean_kaggle.py / 04_integrate.py.")
    if "hf_repo_id" not in repo_df.columns:
        raise ValueError("U repo-level CSV-u nedostaje stupac hf_repo_id. Provjeri 04_integrate.py.")

    row_df["kaggle_row_id"] = row_df["kaggle_row_id"].astype("Int64")

    with sqlite3.connect(DB_PATH) as conn:
        row_df.to_sql("llm_row", conn, if_exists="replace", index=False)
        repo_df.to_sql("llm_repo", conn, if_exists="replace", index=False)

        cur = conn.cursor()

        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_row_kaggle_row_id ON llm_row(kaggle_row_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_row_provider ON llm_row(provider)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_row_context_window ON llm_row(context_window)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_row_hf_repo_id ON llm_row(hf_repo_id)")

        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_repo_hf_repo_id ON llm_repo(hf_repo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_repo_provider ON llm_repo(provider)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_repo_context_window ON llm_repo(context_window)")

        conn.commit()

    print(f"Gotovo. Baza je spremljena u: {DB_PATH}")
    print("Tablice: llm_row (row-level) i llm_repo (repo-level).")


if __name__ == "__main__":
    main()
