import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import os
import random

random.seed(42)
csv_long = os.path.join("datasets", "output", "dos_dataset_long.csv")
csv_hist = os.path.join("datasets", "output", "dos_dataset_histogram_5_ev_cutoff_after_bandgap.csv")
json_folder = Path("datasets/LSODOS")

df_long = pd.read_csv(csv_long)
df_hist = pd.read_csv(csv_hist)

materials = df_hist["material"].unique().tolist()
if len(materials) < 5:
    raise ValueError("Not enough materials to sample 5")
selected = random.sample(materials, 5)

output_dir = os.path.join("plots", "comparison_long_vs_histogram")
os.makedirs(output_dir, exist_ok=True)

for material_name in selected:
    subset = df_long[df_long["material"] == material_name]
    if subset.empty:
        print(f"Skipping {material_name}: not found in long CSV")
        continue

    energies_long = subset["energy"].values
    dos_long = subset["tdos"].values

    if material_name not in df_hist["material"].values:
        print(f"Skipping {material_name}: not found in histogram CSV")
        continue
    row_hist = df_hist[df_hist["material"] == material_name].squeeze()
    energy_columns = [c for c in df_hist.columns if c != "material"]
    energies_hist = np.array([float(c.replace("E=", "").replace("eV", "")) for c in energy_columns])
    dos_hist = row_hist[energy_columns].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].plot(energies_long, dos_long, label="TDOS", lw=1)
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("Total DOS (states/eV)")
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(0, 25)


    axes[1].bar(energies_hist, dos_hist, width=energies_hist[1]-energies_hist[0], align="center", alpha=0.7, edgecolor="blue")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].set_xlim(0, 6)
    axes[1].set_ylim(0, 25)
    fig.tight_layout(rect=[0,0,1,0.95])

    out_file = os.path.join(output_dir, f"{material_name}_comparison_0_to_6_ev.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_file}")
