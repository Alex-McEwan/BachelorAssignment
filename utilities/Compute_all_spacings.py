import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

folder = os.path.join("datasets", "LSODOS")

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]

spacings = []

for file in file_list:
    fpath = os.path.join(folder, file)
    with open(fpath, "r") as f:
        data = json.load(f)

    energies = np.array(data["tdos"]["energies"], dtype=float)
    diffs = np.diff(energies)

    avg_spacing = np.mean(diffs)
    spacings.append((Path(file).stem, avg_spacing))

    print(f"{file}: average ΔE = {avg_spacing:.5f} eV")

df = pd.DataFrame(spacings, columns=["material", "avg_spacing"])
print("\nSummary statistics:")
print(df["avg_spacing"].describe())

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "average_spacings.csv")
df.to_csv(out_file, index=False)
print(f"\nSaved average spacings to: {out_file}")