import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "combinations_full_range", "vacancy_ordered", "halides")
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

for spin in [1, -1]:
    rows = []
    for fname in file_list:
        print(f"Processing {fname} for spin {spin}...")
        fpath = os.path.join(folder, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        tdos = data["tdos_per_site"]

        is_vacancy_ordered = "9" not in tdos  # ✅ old reliable condition


        total_hist = np.zeros(len(bin_centers))

        if is_vacancy_ordered:
            site_range = range(3, 9)   # shifted indices
            print(f"  Detected vacancy-ordered structure.")
        else:
            site_range = range(4, 10)  # normal indices
            print(f"  Detected normal structure.")

        for site_key in site_range:
            energies = np.array(tdos[str(site_key)]["energies"], dtype=float)
            dos = np.array(tdos[str(site_key)]["densities"][str(spin)], dtype=float)
            hist, _ = np.histogram(energies, bins=bin_edges, weights=dos)
            total_hist += hist

        material_name = Path(fname).stem[:-len("_persite")]
        rows.append([material_name] + total_hist.tolist())
        print(f"  Processed {material_name} for spin {spin}.")

    energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
    df = pd.DataFrame(rows, columns=["material"] + energy_cols)

    out_file = os.path.join(output_dir, f"spin{spin}_sites5to10_summed.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
