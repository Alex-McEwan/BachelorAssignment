import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "combinations_full_range","vacancy_ordered", "BBAA")
os.makedirs(output_dir, exist_ok=True)

CONDUCTION_BAND_MINIMUM = 9.80837
LOWEST_AVG_ENERGY_SPACING = 0.00552
LOWEST_GLOBAL_VALENCE_BAND_VALUE = -77.579690
dE = LOWEST_AVG_ENERGY_SPACING

emin = LOWEST_GLOBAL_VALENCE_BAND_VALUE
emax = CONDUCTION_BAND_MINIMUM + 5.0

bin_edges = np.arange(emin, emax + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for site_key in range(4):
    for spin in [1, -1]:
        rows = []
        for fname in file_list:
            fpath = os.path.join(folder, fname)
            with open(fpath, "r") as f:
                data = json.load(f)

            tdos = data["tdos_per_site"]
            material_name = Path(fname).stem[:-len("_persite")]

            is_vacancy_ordered = len(tdos) == 9

            if is_vacancy_ordered:
                if site_key == 1:
                    hist = np.zeros(len(bin_centers))
                    rows.append([material_name] + hist.tolist())
                    continue
                elif site_key > 1:
                    site_index = str(site_key - 1)
                else:
                    site_index = str(site_key)
            else:
                site_index = str(site_key)

            energies = np.array(tdos[site_index]["energies"], dtype=float)
            dos = np.array(tdos[site_index]["densities"][str(spin)], dtype=float)
            hist, _ = np.histogram(energies, bins=bin_edges, weights=dos)

            rows.append([material_name] + hist.tolist())

        energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
        df = pd.DataFrame(rows, columns=["material"] + energy_cols)

        out_file = os.path.join(output_dir, f"site{site_key}_spin{spin}.csv")
        df.to_csv(out_file, index=False)
        print(f"Saved {out_file}")
