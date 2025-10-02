import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ---------------- CONFIG ----------------
folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)

# Which channels to extract: [site_index, spin]
# spin: +1 = up, -1 = down
CHANNELS = [[0, 1], [1, 1], [0, -1], [1, -1]]  # example: B1.up, B2.up, B1.down, B2.down

CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS = 9.80837
LOWEST_AVG_ENERGY_SPACING = 0.00552
dE = LOWEST_AVG_ENERGY_SPACING
emax_global = CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS + 5.0
# ----------------------------------------

# First pass: find global minimum energy
emin_global = float("inf")
file_list = [f for f in os.listdir(folder) if f.endswith(".json")]
for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    energies = np.array(data["tdos"]["energies"], dtype=float)
    emin_global = min(emin_global, energies.min())

# Build energy bins
bin_edges = np.arange(emin_global, emax_global + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

rows = []

# prepare folder to save raw data
raw_output_dir = os.path.join(output_dir, "raw_per_channel")
os.makedirs(raw_output_dir, exist_ok=True)

for fname in file_list:
    fpath = os.path.join(folder, fname)
    print(f"Processing {fname}...")

    with open(fpath, "r") as f:
        data = json.load(f)

    # skip if no per-site DOS or if B2 missing
    if "tdos_per_site" not in data:
        print(f"Skipping {fname} (no tdos_per_site)")
        continue
    tdos_per_site = data["tdos_per_site"]
    if "9" not in tdos_per_site:
        print(f"Skipping {fname} (vacancy ordered, B2 missing)")
        continue

    material_name = Path(fname).stem
    row = [material_name]

    for site_index, spin in CHANNELS:
        site_key = str(site_index)
        spin_key = str(spin)
        site_data = tdos_per_site[site_key]

        dos = np.array(site_data["densities"][spin_key], dtype=float)
        site_energies = np.array(site_data["energies"], dtype=float)

        # ----- save raw data for this channel -----
        raw_df = pd.DataFrame({"Energy(eV)": site_energies, "DOS": dos})
        raw_filename = f"{material_name}_site{site_index}_spin{spin}_raw.csv"
        raw_df.to_csv(os.path.join(raw_output_dir, raw_filename), index=False)
        # -----------------------------------------

        # ----- histogram into global bins -----
        hist, _ = np.histogram(site_energies, bins=bin_edges, weights=dos)
        row.extend(hist)
        hist_df = pd.DataFrame({
        "Energy_center(eV)": bin_centers,
        "DOS_binned": hist
        })
        hist_filename = f"{material_name}_site{site_index}_spin{spin}_hist.csv"
        hist_df.to_csv(os.path.join(raw_output_dir, hist_filename), index=False)

        

    rows.append(row)


rows = [r for r in rows if any(v != 0 for v in r[1:])]

energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
colnames = ["material"] + [f"{site}_{spin}_{e}"
                           for site, spin in CHANNELS
                           for e in energy_cols]

df = pd.DataFrame(rows, columns=colnames)
out_file = os.path.join(output_dir, "dos_dataset_histogram_custom.csv")
df.to_csv(out_file, index=False)
print("Saved to", out_file)
