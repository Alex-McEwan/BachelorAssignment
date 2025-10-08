import pandas as pd
import os

csv_file = os.path.join("datasets", "data_luc", "CombinedHDPinfo_lsodos.csv")

df = pd.read_csv(csv_file)

max_conduction_band_minimum = df["CBM"].max()
max_valence_band_maximum = df["VBM"].max()

print(f"Maximum conduction band minimum (CBM) across all materials: {max_conduction_band_minimum:.5f} eV")
print(f"Maximum valence band maximum (VBM) across all materials: {max_valence_band_maximum:.5f} eV")
