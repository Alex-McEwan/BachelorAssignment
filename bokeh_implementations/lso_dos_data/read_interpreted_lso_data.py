import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

material_name = "1131_CsTlAsCl_lsodos"  



csv_file = os.path.join("datasets", "output", "dos_dataset_interpolated_10_ev_cutoff_after_bandgap.csv")
df = pd.read_csv(csv_file)

if material_name not in df['material'].values:
    print(f"\nERROR: Material '{material_name}' not found in dataset!")
    exit()

material_row = df[df['material'] == material_name].squeeze()  

energy_columns = [col for col in df.columns if col != 'material']

energies = []
dos_values = []

for col in energy_columns:
    energy_str = col.replace("E=", "").replace("eV", "")
    energy = float(energy_str)
    energies.append(energy)
    dos_values.append(material_row[col])

energies = np.array(energies)
dos_values = np.array(dos_values)

plt.figure(figsize=(10, 6))
plt.plot(energies, dos_values, label="Total DOS", linewidth=1.5)

plt.xlabel("Energy (eV)")
plt.ylabel("Total DOS (states/eV)")
plt.title(f"Density of States for {material_name}")
plt.legend()
plt.tight_layout()

output_dir = os.path.join("plots", "lso_interpolated_10_ev_cutoff_after_bandgap")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{material_name}_interpolated_dos_10_ev_cutoff_after_bandgap.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_file}")
plt.close()  
