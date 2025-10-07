import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
import os

# ==== 1. Choose the CSV files you want to combine ====
# Give full paths or relative paths to your files:
files_to_merge = [
    "combinations/BBAA/site0_spin-1.csv",
    "combinations/halides/spin-1_sites5to10_summed.csv"
    # Add more if you want
]

# ==== 2. Load and merge ====
dfs = []
for f in files_to_merge:
    df = pd.read_csv(f)
    dfs.append(df)

# Merge on 'material'
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="material", how="inner")

materials = merged["material"].values

# Drop 'material' column to get features
X = merged.drop(columns=["material"]).values

# ==== 3. Scale and UMAP ====
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(n_neighbors=15, metric="cosine", random_state=42, densmap=True)
X_umap = reducer.fit_transform(X_scaled)

# ==== 4. Create dataframe with UMAP coordinates ====
umap_df = pd.DataFrame({
    "material": materials,
    "x": X_umap[:, 0],
    "y": X_umap[:, 1]
})

print(umap_df.head())
