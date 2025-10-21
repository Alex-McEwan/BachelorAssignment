import pandas as pd
import os

path = os.path.join("datasets", "LSODOS", "data_luc", "CombinedHDPinfo_lsodos.csv")
df = pd.read_csv(path)

subset = df[["comp", "bandgap", "CBM", "magmom_tot_lobster", "cond_type"]].copy()


#print all of the cond types
cond_types = subset["cond_type"].unique()
print("Unique cond_type values:")
for ct in cond_types:
    print(f"- {ct}")
    #print the counts for each cond type
    print(f"  Count: {subset[subset['cond_type'] == ct].shape[0]}")


subset = subset.rename(columns={"comp": "material"})

subset["material"] = subset["material"].astype(str) + "_lsodos"

output_dir = os.path.join("datasets", "output")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "material_bandgap.csv")
subset.to_csv(output_path, index=False)

print(f"Bandgap CSV saved to: {output_path}")