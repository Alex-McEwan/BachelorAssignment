import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = os.path.join("datasets", "output", "average_spacings.csv")
df = pd.read_csv(csv_path)

min_spacing = df["avg_spacing"].min() 
min_spacing_material = df[df["avg_spacing"] == min_spacing]["material"].iloc[0]  

print(f"Lowest spacing: {min_spacing:.5f} eV (material: {min_spacing_material})")

plt.figure(figsize=(8, 5))
counts, bins, patches = plt.hist(df["avg_spacing"], bins=80, edgecolor="black")

for i in range(len(counts)):
    if counts[i] > 0:
        label = f"{bins[i]:.4f}–{bins[i+1]:.4f}"
        plt.text(
            (bins[i] + bins[i+1]) / 2,
            counts[i],
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7
        )

plt.xlabel("Average ΔE (eV)")
plt.ylabel("Count")
plt.title("Distribution of average energy spacings across materials")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_file = os.path.join("datasets", "output", "average_spacings_hist_bins.png")
plt.savefig(out_file)
plt.show()
