import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
import os

folder = os.path.join("datasets", "LSODOS")


emin_global, emax_global = float("inf"), float("-inf")

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

file_list = file_list[:500]  

for fname in file_list:
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    energies = np.array(data["tdos"]["energies"], dtype=float)
    emin_global = min(emin_global, energies.min())
    emax_global = max(emax_global, energies.max())

print(f"Global energy range: {emin_global:.3f} eV → {emax_global:.3f} eV")

dE = 0.001  
energy_grid = np.arange(emin_global, emax_global + dE, dE)

rows = []

for fname in file_list:
    if not fname.endswith(".json"):
        continue
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

colnames = ["material"] + [f"E={e:.3f}eV" for e in energy_grid]
df = pd.DataFrame(rows, columns=colnames)

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "dos_dataset_interpolated.csv")
df.to_csv(out_file, index=False)

print("Final shape:", df.shape)
print("Saved to:", out_file)
