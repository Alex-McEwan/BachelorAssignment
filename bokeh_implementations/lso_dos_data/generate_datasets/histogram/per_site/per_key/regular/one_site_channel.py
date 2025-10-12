import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ----------- USER INPUT -----------
SITE_KEY = "1"   # e.g. "1"
SPIN = "-1"      # e.g. "-1"
# ----------------------------------

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "per_site")
os.makedirs(output_dir, exist_ok=True)

CONDUCTION_BAND_MINIMUM = 9.80837
VALENCE_BAND_MAXIMUM = -0.00554
LOWEST_AVG_ENERGY_SPACING = 0.00552

dE = LOWEST_AVG_ENERGY_SPACING
emin = VALENCE_BAND_MAXIMUM - 5.0
emax = CONDUCTION_BAND_MINIMUM + 5.0

bin_edges = np.arange(emin, emax + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

rows = []

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    # skip files missing key 9
    if "tdos_per_site" not in data or "9" not in data["tdos_per_site"]:
        print(f"Skipping {fname} (no key 9)")
        continue

    tdos = data["tdos_per_site"]

    print(f"Processing {fname} ...")
    site_data = tdos[SITE_KEY]
    energies = np.array(site_data["energies"], dtype=float)
    dos = np.array(site_data["densities"][SPIN], dtype=float)

    hist, _ = np.histogram(energies, bins=bin_edges, weights=dos)

    material_name = Path(fname).stem[:-len("_persite")]
    rows.append([material_name] + hist.tolist())

# make dataframe
energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
df = pd.DataFrame(rows, columns=["material"] + energy_cols)

out_file = os.path.join(output_dir, f"site{SITE_KEY}_spin{SPIN}.csv")
df.to_csv(out_file, index=False)
print(f"\nSaved {out_file}")
