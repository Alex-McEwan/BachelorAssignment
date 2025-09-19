import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import os
# Load dataset
csv_file = os.path.join("datasets", "output", "dos_dataset_interpolated.csv")
df = pd.read_csv(csv_file)

# Extract material names
materials = df["material"].values

# Extract DOS matrix (all energy columns)
energy_columns = [col for col in df.columns if col != "material"]
X = df[energy_columns].to_numpy()

# Standardize DOS values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run UMAP
print("started UMAP")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

# Save result in DataFrame
umap_df = pd.DataFrame({
    "material": materials,
    "x": X_umap[:, 0],
    "y": X_umap[:, 1],
})

# --- Matplotlib scatterplot ---
plt.figure(figsize=(8, 6))
plt.scatter(umap_df["x"], umap_df["y"], s=30, alpha=0.7)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP projection of DOS data")
plt.show()
