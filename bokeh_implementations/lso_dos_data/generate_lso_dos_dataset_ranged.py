import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
import os

folder = os.path.join("datasets", "LSODOS")

dE = 0.01   
rows = []

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    energies = np.array(data["tdos"]["energies"], dtype=float)
    efermi = float(data["tdos"]["efermi"])
    dens_up = np.array(data["tdos"]["densities"]["1"], dtype=float)
    dens_dn = np.array(data["tdos"]["densities"]["-1"], dtype=float)
    dos = dens_up + dens_dn


    #todo increase range to 10 eV around fermi level
    emin = efermi - 5.0
    emax = efermi + 5.0
    energy_grid = np.arange(emin, emax + dE, dE)

    interp = interp1d(  
        energies, dos,
        bounds_error=False, 
        fill_value=0.0,
        kind="nearest"
    )
    dos_resampled = interp(energy_grid)

    material_name = Path(fname).stem
    rows.append([material_name, *dos_resampled])
    print(f"Processed {material_name}, range: {emin:.2f} → {emax:.2f} eV, shape: {dos_resampled.shape}")

colnames = ["material"] + [f"E={e:.3f}eV" for e in energy_grid]
df = pd.DataFrame(rows, columns=colnames)

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "dos_dataset_interpolated_5ev_around_fermi.csv")
df.to_csv(out_file, index=False)

print("Final shape:", df.shape)
print("Saved to:", out_file)
