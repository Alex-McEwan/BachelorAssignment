import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# === Configuration ===
base_dir = os.path.join("datasets", "output", "combinations_full_range", "vacancy_ordered")
combo1 = [
    os.path.join(base_dir, "BBAA", "site0_spin1.csv"),   # B1.up
    os.path.join(base_dir, "BBAA", "site1_spin1.csv"),   # B2.up
    os.path.join(base_dir, "BBAA", "site0_spin-1.csv"),  # B1.down
    os.path.join(base_dir, "BBAA", "site1_spin-1.csv")   # B2.down
]

output_dir = os.path.join("plots", "B_sites_spin_summed")
os.makedirs(output_dir, exist_ok=True)

random.seed(42)

# === Load all four CSVs ===
dfs = [pd.read_csv(f) for f in combo1]
for i, f in enumerate(combo1):
    print(f"Loaded {f} with {dfs[i].shape[0]} rows and {dfs[i].shape[1]} columns")

# === Rename columns for clarity ===
for i, df in enumerate(dfs):
    prefix = os.path.splitext(os.path.basename(combo1[i]))[0]
    df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "material"}, inplace=True)

# === Merge all by material ===
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="material", how="inner")

materials = merged["material"].unique().tolist()
if len(materials) < 5:
    raise ValueError("Not enough materials to sample 5")
selected = random.sample(materials, 5)

# === Prepare energy columns ===
energy_cols = [c for c in merged.columns if c.endswith("E=0.000eV")]  # pick pattern dynamically
all_energy_cols = [c for c in merged.columns if "E=" in c]
energies = np.array([float(c.split("_E=")[-1].replace("eV", "")) for c in all_energy_cols if "site0_spin1" in c])

# === Plot loop ===
for material in selected:
    row = merged[merged["material"] == material].squeeze()

    # Sum spins for each site
    site0_up = row[[c for c in merged.columns if "site0_spin1" in c]].to_numpy(dtype=float)
    site0_down = row[[c for c in merged.columns if "site0_spin-1" in c]].to_numpy(dtype=float)
    site1_up = row[[c for c in merged.columns if "site1_spin1" in c]].to_numpy(dtype=float)
    site1_down = row[[c for c in merged.columns if "site1_spin-1" in c]].to_numpy(dtype=float)

    dos_site0 = site0_up + site0_down
    dos_site1 = site1_up + site1_down

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].plot(energies, dos_site0, lw=1, label="B1 (site0)")
    axes[1].plot(energies, dos_site1, lw=1, label="B2 (site1)")

    for ax in axes:
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS (states/eV)")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)

    fig.suptitle(material)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = os.path.join(output_dir, f"{material}_Bsites_spin_summed.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_file}")
