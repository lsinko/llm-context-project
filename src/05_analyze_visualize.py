import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROW_LEVEL_CSV = "data/processed/merged_llm_data.csv"
REPO_LEVEL_CSV = "data/processed/merged_llm_data_repo_level.csv"
FIG_DIR = "reports/figures"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True, ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return s * 0
    return (s - mu) / sigma


def save_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    logx: bool = False,
    logy: bool = False,
):
    if x not in df.columns or y not in df.columns:
        return

    tmp = df[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty:
        return

    plt.figure()
    plt.scatter(tmp[x], tmp[y], alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()


def save_hist_by_provider(
    df: pd.DataFrame,
    value_col: str,
    provider_col: str,
    title: str,
    xlabel: str,
    filename: str,
    max_providers: int = 8,
):
    if value_col not in df.columns or provider_col not in df.columns:
        return

    tmp = df[[provider_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp[provider_col] = tmp[provider_col].astype(str).str.strip()
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return

    counts = tmp[provider_col].value_counts()
    providers = counts.head(max_providers).index.tolist()
    tmp = tmp[tmp[provider_col].isin(providers)].copy()

    plt.figure()
    for p in providers:
        vals = tmp.loc[tmp[provider_col] == p, value_col].dropna().values
        if len(vals) > 0:
            plt.hist(vals, bins=25, alpha=0.5, label=p)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frekvencija")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()


def save_corr_heatmap(df: pd.DataFrame, cols: list[str], title: str, filename: str):
    usable = [c for c in cols if c in df.columns]
    if len(usable) < 2:
        return

    tmp = df[usable].copy()
    for c in usable:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return

    corr = tmp.corr(numeric_only=True)

    plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(usable)), usable, rotation=45, ha="right")
    plt.yticks(range(len(usable)), usable)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()


def main():
    ensure_dirs()

    if not os.path.exists(ROW_LEVEL_CSV):
        raise FileNotFoundError(f"Nedostaje {ROW_LEVEL_CSV}. Prvo pokreni 04_integrate.py.")
    if not os.path.exists(REPO_LEVEL_CSV):
        raise FileNotFoundError(f"Nedostaje {REPO_LEVEL_CSV}. Prvo pokreni 04_integrate.py.")

    row_df = pd.read_csv(ROW_LEVEL_CSV)
    repo_df = pd.read_csv(REPO_LEVEL_CSV)

    row_numeric = [
        "context_window",
        "latency_sec",
        "speed_tokens_per_sec",
        "benchmark_mmlu",
        "benchmark_chatbot_arena",
    ]
    repo_numeric = [
        "context_window",
        "latency_sec",
        "speed_tokens_per_sec",
        "benchmark_mmlu",
        "benchmark_chatbot_arena",
        "hf_downloads",
        "hf_likes",
        "hf_downloads_all_time",
        "n_kaggle_rows",
    ]

    row_df = to_numeric(row_df, row_numeric)
    repo_df = to_numeric(repo_df, repo_numeric)

    save_scatter(
        df=row_df,
        x="context_window",
        y="latency_sec",
        title="Odnos veličine konteksta i latencije (row-level)",
        xlabel="context_window",
        ylabel="latency_sec",
        filename="01_context_vs_latency_row.png",
        logx=True,
        logy=False,
    )

    save_scatter(
        df=row_df,
        x="context_window",
        y="speed_tokens_per_sec",
        title="Odnos veličine konteksta i brzine generiranja (row-level)",
        xlabel="context_window",
        ylabel="speed_tokens_per_sec",
        filename="02_context_vs_speed_row.png",
        logx=True,
        logy=False,
    )

    save_hist_by_provider(
        df=row_df,
        value_col="context_window",
        provider_col="provider",
        title="Raspodjela veličine konteksta po provideru (row-level, log skala)",
        xlabel="context_window (log)",
        filename="03_context_hist_by_provider.png",
        max_providers=8,
    )

    save_scatter(
        df=repo_df,
        x="context_window",
        y="hf_downloads",
        title="Odnos veličine konteksta i popularnosti (repo-level)",
        xlabel="context_window",
        ylabel="hf_downloads",
        filename="04_context_vs_downloads_repo.png",
        logx=True,
        logy=True,
    )

    repo_norm = repo_df.copy()
    for c in ["context_window", "latency_sec", "speed_tokens_per_sec", "benchmark_mmlu", "hf_downloads", "hf_likes"]:
        if c in repo_norm.columns:
            repo_norm[c + "_z"] = zscore(repo_norm[c])

    save_corr_heatmap(
        df=repo_norm,
        cols=[c for c in ["context_window_z", "latency_sec_z", "speed_tokens_per_sec_z", "benchmark_mmlu_z", "hf_downloads_z", "hf_likes_z"] if c in repo_norm.columns],
        title="Korelacija normaliziranih varijabli (repo-level, Z-score)",
        filename="05_repo_level_corr_heatmap.png",
    )

    out_norm_csv = "data/processed/merged_llm_data_repo_level_normalized.csv"
    repo_norm.to_csv(out_norm_csv, index=False, encoding="utf-8-sig")

    print("Gotovo. Grafovi su spremljeni u:", FIG_DIR)
    print("Repo-level normalizirani CSV je spremljen u:", out_norm_csv)


if __name__ == "__main__":
    main()
