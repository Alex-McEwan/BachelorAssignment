import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

material_name = "1040_CsNiBaCl_lsodos"
csv_file = os.path.join("datasets", "output", "dos_dataset_histogram_5_ev_cutoff_after_bandgap.csv")

df = pd.read_csv(csv_file)

if material_name not in df['material'].values:
    print(f"\nERROR: Material '{material_name}' not found in dataset!")
    exit()

material_row = df[df['material'] == material_name].squeeze()

energy_columns = [col for col in df.columns if col != 'material']
energies = np.array([float(c.replace("E=", "").replace("eV", "")) for c in energy_columns])
dos_values = material_row[energy_columns].to_numpy(dtype=float)

plt.figure(figsize=(10, 6))
plt.bar(energies, dos_values, width=energies[1] - energies[0], align='center', alpha=0.7, edgecolor='blue')

plt.xlabel("Energy (eV)")
plt.ylabel("Total DOS (states/eV)")
plt.title(f"Density of States (Histogram) for {material_name}")
plt.tight_layout()

output_dir = os.path.join("plots", "lso_histogram_5_ev_cutoff_after_bandgap")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{material_name}_histogram_dos_5_ev_cutoff_after_bandgap.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_file}")
plt.close()
