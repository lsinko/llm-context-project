import pandas as pd

df = pd.read_csv("data/raw/llm_comparison_dataset.csv")
print("CSV READ OK")

print(df.shape)
print(df.columns.tolist())
print(df.head(50).to_string(index=False))


