import pandas as pd
import os

csv_file = os.path.join("datasets", "data_luc", "CombinedHDPinfo_lsodos.csv")

df = pd.read_csv(csv_file)

max_bandgap = df["bandgap"].max()

print("Largest bandgap:", max_bandgap, "eV")
