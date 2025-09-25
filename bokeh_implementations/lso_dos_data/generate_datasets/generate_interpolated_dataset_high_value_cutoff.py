import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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

print(f"Global energy range: {emin_global:.3f} eV → {emax_global:.3f} eV")

LOWEST_AVG_ENERGY_SPACING = 0.00552


dE = LOWEST_AVG_ENERGY_SPACING
energy_grid = np.arange(emin_global, emax_global + dE, dE)

rows = []

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    energies = np.array(data["tdos"]["energies"], dtype=float)
    dens_up = np.array(data["tdos"]["densities"]["1"], dtype=float)
    dens_dn = np.array(data["tdos"]["densities"]["-1"], dtype=float)
    dos = dens_up + dens_dn

    interp = interp1d(
        energies, dos,
        bounds_error=False,
        fill_value=0.0,
        kind="nearest"
    )
    dos_resampled = interp(energy_grid)

    material_name = Path(fname).stem
    rows.append([material_name, *dos_resampled])
    print(f"Processed {material_name}, shape: {dos_resampled.shape}")

colnames = ["material"] + [f"E={e:.3f}eV" for e in energy_grid]
df = pd.DataFrame(rows, columns=colnames)

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "dos_dataset_interpolated_5_ev_cutoff_after_bandgap.csv")
df.to_csv(out_file, index=False)

print("Final shape:", df.shape)
print("Saved to:", out_file)
