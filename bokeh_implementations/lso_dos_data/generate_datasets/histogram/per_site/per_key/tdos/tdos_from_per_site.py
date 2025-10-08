import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")
output_dir = os.path.join("datasets", "output", "combinations_full_range", "tdos")
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

rows_up = []   # spin = 1
rows_down = [] # spin = -1

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    print(f"Processing {fname} ...")
    tdos = data["tdos"]
    energies = np.array(tdos["energies"], dtype=float)

    # spin up (1)
    dos_up = np.array(tdos["densities"]["1"], dtype=float)
    hist_up, _ = np.histogram(energies, bins=bin_edges, weights=dos_up)
    rows_up.append([Path(fname).stem[:-len("_persite")]] + hist_up.tolist())

    # spin down (-1)
    dos_down = np.array(tdos["densities"]["-1"], dtype=float)
    hist_down, _ = np.histogram(energies, bins=bin_edges, weights=dos_down)
    rows_down.append([Path(fname).stem[:-len("_persite")]] + hist_down.tolist())

energy_cols = [f"E={e:.3f}eV" for e in bin_centers]

if rows_up:
    df_up = pd.DataFrame(rows_up, columns=["material"] + energy_cols)
    out_up = os.path.join(output_dir, "tdos_spin1.csv")
    df_up.to_csv(out_up, index=False)
    print(f"Saved {out_up}")

if rows_down:
    df_down = pd.DataFrame(rows_down, columns=["material"] + energy_cols)
    out_down = os.path.join(output_dir, "tdos_spin-1.csv")
    df_down.to_csv(out_down, index=False)
    print(f"Saved {out_down}")



