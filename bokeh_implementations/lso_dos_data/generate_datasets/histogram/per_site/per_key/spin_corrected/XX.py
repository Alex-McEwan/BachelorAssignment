import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "combinations_full_range", "spin_corrected", "halides")
os.makedirs(output_dir, exist_ok=True)

magmom_csv = os.path.join("datasets", "output", "material_bandgap.csv")
magmom_df = pd.read_csv(magmom_csv)
magmom_dict = dict(zip(magmom_df["material"], magmom_df["magmom_tot_lobster"]))

CONDUCTION_BAND_MINIMUM = 9.80837
VALENCE_BAND_MAXIMUM = -0.00554
LOWEST_AVG_ENERGY_SPACING = 0.00552

dE = LOWEST_AVG_ENERGY_SPACING
emin = -77.579690
emax = CONDUCTION_BAND_MINIMUM + 5.0

bin_edges = np.arange(emin, emax + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for spin in [1, -1]:
    rows = []
    for fname in file_list:
        fpath = os.path.join(folder, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        if "tdos_per_site" not in data or "9" not in data["tdos_per_site"]:
            print(f"Skipping {fname} (no key 9)")
            continue

        material_name = Path(fname).stem[:-len("_persite")]
        magmom = magmom_dict[material_name]  # will raise KeyError if not found

        total_hist = np.zeros(len(bin_centers))

        for site_key in range(5, 10):
            site_data = data["tdos_per_site"][str(site_key)]
            energies = np.array(site_data["energies"], dtype=float)
            dos = np.array(site_data["densities"][str(spin)], dtype=float)

            if magmom < 0:
                dos = -dos

            hist, _ = np.histogram(energies, bins=bin_edges, weights=dos)
            total_hist += hist

        rows.append([material_name] + total_hist.tolist())

    energy_cols = [f"E={e:.3f}eV" for e in bin_centers]
    df = pd.DataFrame(rows, columns=["material"] + energy_cols)

    out_file = os.path.join(output_dir, f"spin{spin}_sites5to10_summed.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
