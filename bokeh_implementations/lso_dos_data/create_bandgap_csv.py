import pandas as pd
import os

path = os.path.join("datasets", "data_luc", "CombinedHDPinfo_lsodos.csv")

df = pd.read_csv(path)

subset = df[["comp", "bandgap"]]

subset = subset.rename(columns={"comp": "material"})



#TODO ensure that the material names match those in the dos dataset exactly
output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)

subset.to_csv(os.path.join(output_dir, "material_bandgap.csv"), index=False)


