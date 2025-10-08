import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "combinations_full_range", "BBAA")
os.makedirs(output_dir, exist_ok=True)

CONDUCTION_BAND_MINIMUM = 9.80837
VALENCE_BAND_MAXIMUM = -0.00554
LOWEST_AVG_ENERGY_SPACING = 0.00552

dE = LOWEST_AVG_ENERGY_SPACING
emin = -77.579690
emax = CONDUCTION_BAND_MINIMUM + 5.0

bin_edges = np.arange(emin, emax + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for site_key in range(5):
    for spin in [1, -1]:
        rows = []
        for fname in file_list:
            fpath = os.path.join(folder, fname)
            with open(fpath, "r") as f:
                data = json.load(f)

            # skip files missing key 9
            if "tdos_per_site" not in data or "9" not in data["tdos_per_site"]:
                print(f"Skipping {fname} (no key 9)")
                continue

            tdos = data["tdos_per_site"]

            print(f"Processing {fname} for site {site_key}, spin {spin}...")
            site_data = tdos[str(site_key)]
            energies = np.array(site_data["energies"], dtype=float)
            dos = np.array(site_data["densities"][str(spin)], dtype=float)

            hist, _ = np.histogram(energies, bins=bin_edges, weights=dos)

            material_name = Path(fname).stem[:-len("_persite")]
            rows.append([material_name] + hist.tolist())

        # make dataframe for this site+spin
        energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
        df = pd.DataFrame(rows, columns=["material"] + energy_cols)

        out_file = os.path.join(output_dir, f"site{site_key}_spin{spin}.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved {out_file}")
    







    