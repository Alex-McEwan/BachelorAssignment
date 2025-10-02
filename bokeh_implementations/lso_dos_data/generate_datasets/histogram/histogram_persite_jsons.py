import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ---------------- CONFIG ----------------
folder = os.path.join("datasets", "LSODOS")
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

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    tdos_per_site = data["tdos_per_site"]
    if "9" not in tdos_per_site:
        print(f"Skipping {fname} (vacancy ordered, B2 missing)")
        continue


    material_name = Path(fname).stem
    row = [material_name]

    for site_index, spin in CHANNELS:
        if str(site_index) in tdos_per_site:
            site_data = tdos_per_site[str(site_index)]
            if "densities" in site_data and str(spin) in site_data["densities"]:
                dos = np.array(site_data["densities"][str(spin)], dtype=float)
                site_energies = np.array(site_data["energies"], dtype=float)
            else:
                raise ValueError(f"Missing densities or spin {spin} for site {site_index} in file {fname}")
        else:
            raise ValueError(f"Site index {site_index} not found in file {fname}")
        hist, _ = np.histogram(site_energies, bins=bin_edges, weights=dos)
        row.extend(hist)

    rows.append(row)

rows = [r for r in rows if any(v != 0 for v in r)]

energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
colnames = ["material"] + [f"{site}_{spin}_{e}"
                           for site, spin in CHANNELS
                           for e in energy_cols]

df = pd.DataFrame(rows, columns=colnames)
out_file = os.path.join(output_dir, "dos_dataset_histogram_custom.csv")
df.to_csv(out_file, index=False)
print("Saved to", out_file)
