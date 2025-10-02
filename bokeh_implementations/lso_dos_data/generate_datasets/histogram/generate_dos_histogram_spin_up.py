import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "LSODOS")

emin_global = float("inf")
file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    energies = np.array(data["tdos"]["energies"], dtype=float)
    emin_global = min(emin_global, energies.min())

CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS = 9.80837
emax_global = CONDUCTION_BAND_MINIMUM_ACROSS_ALL_MATERIALS + 5.0

LOWEST_AVG_ENERGY_SPACING = 0.00552
dE = LOWEST_AVG_ENERGY_SPACING

bin_edges = np.arange(emin_global, emax_global + dE, dE)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

rows = []

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    energies = np.array(data["tdos"]["energies"], dtype=float)
    dens_up = np.array(data["tdos"]["densities"]["1"], dtype=float)

    dos_binned, _ = np.histogram(energies, bins=bin_edges, weights=dens_up)

    material_name = Path(fname).stem
    rows.append([material_name, *dos_binned])

colnames = ["material"] + [f"E={e:.3f}eV" for e in bin_centers]
df = pd.DataFrame(rows, columns=colnames)

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "dos_dataset_histogram_5_ev_cutoff_after_bandgap_spin_up.csv")
df.to_csv(out_file, index=False)
