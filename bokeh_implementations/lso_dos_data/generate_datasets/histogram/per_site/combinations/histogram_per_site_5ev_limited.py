import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)

CHANNELS = [[0, 1], [1, 1], [0, -1], [1, -1]]  

CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS = 9.80837
LOWEST_AVG_ENERGY_SPACING = 0.00552
VALENCE_BAND_MAXIMUM_ACROSS_ALL_MATERIALS = -0.00554
dE = LOWEST_AVG_ENERGY_SPACING
emax_global = CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS + 5.0

emin_global = VALENCE_BAND_MAXIMUM_ACROSS_ALL_MATERIALS - 5.0

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]


bin_edges = np.arange(emin_global, emax_global + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

rows = []

raw_output_dir = os.path.join(output_dir, "raw_per_channel")
one_site_hist_dir = os.path.join(output_dir, "one_site_histograms")
os.makedirs(raw_output_dir, exist_ok=True)
os.makedirs(one_site_hist_dir, exist_ok=True)

for fname in file_list:
    fpath = os.path.join(folder, fname)
    print(f"Processing {fname}...")

    with open(fpath, "r") as f:
        data = json.load(f)

    if "tdos_per_site" not in data:
        print(f"Skipping {fname} (no tdos_per_site)")
        continue
    tdos_per_site = data["tdos_per_site"]
    if "9" not in tdos_per_site:
        print(f"Skipping {fname} (vacancy ordered, B2 missing)")
        continue

    material_name = Path(fname).stem[:-len("_persite")]

    row = [material_name]

    for site_index, spin in CHANNELS:
        site_key = str(site_index)
        spin_key = str(spin)
        site_data = tdos_per_site[site_key]

        dos = np.array(site_data["densities"][spin_key], dtype=float)
        site_energies = np.array(site_data["energies"], dtype=float)

        raw_df = pd.DataFrame({"Energy(eV)": site_energies, "DOS": dos})
        raw_filename = f"{material_name}_site{site_index}_spin{spin}_raw.csv"
        raw_df.to_csv(os.path.join(raw_output_dir, raw_filename), index=False)

        hist, _ = np.histogram(site_energies, bins=bin_edges, weights=dos)
        row.extend(hist)
        hist_df = pd.DataFrame({
        "Energy_center(eV)": bin_centers,
        "DOS_binned": hist
        })
        hist_filename = f"{material_name}_site{site_index}_spin{spin}_hist.csv"
        hist_df.to_csv(os.path.join(one_site_hist_dir, hist_filename), index=False)

        

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
