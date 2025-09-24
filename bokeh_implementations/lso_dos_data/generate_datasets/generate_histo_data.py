import os
import json
import pandas as pd
from pymatgen.electronic_structure.dos import LobsterCompleteDos

INPUT_FOLDER = os.path.join("datasets", "completedosjsons_lsodos")
OUTPUT_DIR = os.path.join("datasets", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_E = -5.0
MAX_E = 20.0
N_BINS = 256
FP_TYPE = "summed_pdos"

CSV_OUT = os.path.join(OUTPUT_DIR, f"dos_fingerprints_{FP_TYPE}_{N_BINS}bins.csv")

rows = []
for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(INPUT_FOLDER, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    try:
        dos = LobsterCompleteDos.from_dict(data)
        fp = dos.get_dos_fp(
            fp_type=FP_TYPE,
            max_e=MAX_E,
            n_bins=N_BINS,
            normalize=True
        )
    except Exception as e:
        print(f"Skipping {fname}, error: {e}")
        continue
    material = os.path.splitext(fname)[0]
    rows.append([material, *fp.densities])
    print(f"Processed {material}, fp length={len(fp.densities)}")

colnames = ["material"] + [f"bin{i}" for i in range(N_BINS)]
df = pd.DataFrame(rows, columns=colnames)
df.to_csv(CSV_OUT, index=False)
print(f"Saved fingerprints to {CSV_OUT}, shape={df.shape}")
