import os
import pandas as pd

IN_PATH = "data/raw/llm_comparison_dataset.csv"
OUT_PATH = "data/processed/kaggle_clean.csv"

df = pd.read_csv(IN_PATH)

# stabilan ID izvornog retka
df.insert(0, "kaggle_row_id", range(len(df)))

df = df.rename(columns={
    "Model": "model_name",
    "Provider": "provider",
    "Context Window": "context_window",
    "Speed (tokens/sec)": "speed_tokens_per_sec",
    "Latency (sec)": "latency_sec",
    "Benchmark (MMLU)": "benchmark_mmlu",
    "Benchmark (Chatbot Arena)": "benchmark_chatbot_arena",
    "Open-Source": "open_source",
    "Price / Million Tokens": "price_per_million_tokens",
    "Training Dataset Size": "training_dataset_size",
    "Compute Power": "compute_power",
    "Energy Efficiency": "energy_efficiency",
    "Quality Rating": "quality_rating",
    "Speed Rating": "speed_rating",
    "Price Rating": "price_rating",
})

num_cols = [
    "context_window", "speed_tokens_per_sec", "latency_sec",
    "benchmark_mmlu", "benchmark_chatbot_arena",
    "price_per_million_tokens", "training_dataset_size",
    "compute_power", "energy_efficiency",
    "quality_rating", "speed_rating", "price_rating",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Saved:", OUT_PATH, "rows=", len(df))
