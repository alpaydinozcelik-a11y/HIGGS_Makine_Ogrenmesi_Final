import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path("data/HIGGS.csv.gz")
output_path = Path("data/higgs_100k.csv")

columns = ["target"] + [f"feature_{i}" for i in range(1, 29)]

sample_size = 100_000
chunk_size = 200_000
random_state = 42

rng = np.random.default_rng(random_state)

reservoir = pd.DataFrame()
processed_rows = 0

print("Rastgele 100.000 örnek seçimi başladı...")

for chunk in pd.read_csv(
    input_path,
    compression="gzip",
    header=None,
    names=columns,
    chunksize=chunk_size
):
    chunk["_random_key"] = rng.random(len(chunk))

    reservoir = pd.concat([reservoir, chunk], ignore_index=True)

    if len(reservoir) > sample_size:
        reservoir = reservoir.nsmallest(sample_size, "_random_key").reset_index(drop=True)

    processed_rows += len(chunk)
    print(f"{processed_rows:,} satır tarandı...")

sample_df = (
    reservoir
    .drop(columns=["_random_key"])
    .sample(frac=1, random_state=random_state)
    .reset_index(drop=True)
)

sample_df["target"] = sample_df["target"].astype(int)

sample_df.to_csv(output_path, index=False)

print("\nRastgele örneklem tamamlandı.")
print("Kaydedilen dosya:", output_path)
print("Veri boyutu:", sample_df.shape)
print("\nSınıf dağılımı:")
print(sample_df["target"].value_counts())