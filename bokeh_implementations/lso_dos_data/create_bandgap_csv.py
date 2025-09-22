import pandas as pd
import os

path = os.path.join("datasets", "data_luc", "CombinedHDPinfo_lsodos.csv")
df = pd.read_csv(path)

subset = df[["comp", "bandgap"]].copy()

subset = subset.rename(columns={"comp": "material"})

subset["material"] = subset["material"].astype(str) + "_lsodos"

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "material_bandgap.csv")
subset.to_csv(output_path, index=False)

print(f"Bandgap CSV saved to: {output_path}")