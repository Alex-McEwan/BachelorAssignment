import json
import numpy as np
import pandas as pd
from pathlib import Path
import os

folder = Path(os.path.join("datasets", "LSODOS"))
records = []

for file in folder.glob("*.json"):
    with open(file, "r") as f:
        data = json.load(f)

    energies = np.array(data["tdos"]["energies"], dtype=float)
    dens_up = np.array(data["tdos"]["densities"]["1"], dtype=float)
    dens_dn = np.array(data["tdos"]["densities"]["-1"], dtype=float)
    dos = dens_up + dens_dn

    for energy, dos in zip(energies, dos):
        records.append({"material": file.stem, "energy": energy, "tdos": dos})

df = pd.DataFrame(records)

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "dos_dataset_long.csv"), index=False)

print(df.head())
