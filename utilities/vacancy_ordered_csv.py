import os
import json
import pandas as pd

base = os.path.join("datasets", "lsodos_persitejsons_250930")
output_csv = os.path.join("datasets", "output", "vacancy_ordered_labels.csv")

records = []
files = [f for f in os.listdir(base) if f.endswith(".json")]

for fname in files:
    path = os.path.join(base, fname)
    material = os.path.splitext(fname)[0]
    if material.endswith("_persite"):
        material = material[:-8]  # remove "_persite" (8 characters)
    is_vacancy_ordered = False
    try:
        with open(path) as f:
            data = json.load(f)
        tdos = data.get("tdos_per_site", {})
        if "9" not in tdos:
            is_vacancy_ordered = True
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue
    records.append({"material": material, "vacancy_ordered": is_vacancy_ordered})

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print(f"Saved {len(df)} entries to {output_csv}")
print(f"Number of vacancy ordered HDPs: {df['vacancy_ordered'].sum()} / {len(df)}")
