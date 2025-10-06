import json
import numpy as np
import os

folder = os.path.join("datasets", "lsodos_persitejsons_250930")

emin = {str(i): float("inf") for i in range(10)}
emax = {str(i): -float("inf") for i in range(10)}

file_list = [f for f in os.listdir(folder) if f.endswith(".json")]
for fname in file_list:
    fpath = os.path.join(folder, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    if "tdos_per_site" not in data or "9" not in data["tdos_per_site"]:
        print(f"Skipping {fname} (no key 9)")
        continue
    tdos = data["tdos_per_site"]
    for k in emin.keys():
        e = np.array(tdos[k]["energies"], dtype=float)
        emin[k] = min(emin[k], e.min())
        emax[k] = max(emax[k], e.max())

for k in emin.keys():
    if emin[k] != float("inf"):
        print(f"site {k}: min {emin[k]:.6f} eV , max {emax[k]:.6f} eV")
    else:
        print(f"site {k}: no data")
