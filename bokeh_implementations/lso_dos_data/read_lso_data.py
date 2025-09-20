import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

csv_file = os.path.join("datasets", "output", "dos_dataset_long.csv")
json_folder = Path("datasets/LSODOS")  
material_name = "1002_CsNaPrCl_lsodos"  

df = pd.read_csv(csv_file)

subset = df[df["material"] == material_name]
if subset.empty:
    raise ValueError(f"Material {material_name} not found in {csv_file}")

energies = subset["energy"].values
dos = subset["tdos"].values

json_file = json_folder / f"{material_name}.json"
if not json_file.exists():
    raise FileNotFoundError(f"JSON file {json_file} not found")

with open(json_file, "r") as f:
    data = json.load(f)

efermi = float(data["tdos"]["efermi"])
print(f"Plotting {material_name} with Fermi level at {efermi:.3f} eV")

plt.figure(figsize=(6, 4))
plt.plot(energies, dos, label="TDOS")
plt.axvline(efermi, ls="--", color="k", alpha=0.7, label="$E_F$")
plt.xlabel("Energy (eV)")
plt.ylabel("Total DOS (states/eV)")
plt.title(f"Total DOS for {material_name}")
plt.legend()
plt.tight_layout()

output_dir = os.path.join("plots", "lso_original")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{material_name}_dos.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_file}")
plt.close()  
