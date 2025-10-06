import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_TO_CHECK = os.path.join("datasets", "output", "per_site", "site0_spin-1.csv")
SITE_KEY = "0"
SPIN = "-1"
JSON_FOLDER = os.path.join("datasets", "lsodos_persitejsons_250930")
PLOT_DIR = os.path.join("datasets", "output", "plots", "check_csv")
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES = 5

random.seed(RANDOM_SEED)

df_hist = pd.read_csv(CSV_TO_CHECK)
materials = df_hist["material"].tolist()
energy_cols = df_hist.columns[1:]
bin_centers = [float(col.split("=")[1].replace("eV", "")) for col in energy_cols]

sample_materials = random.sample(materials, min(N_SAMPLES, len(materials)))

for material in sample_materials:
    row = df_hist[df_hist["material"] == material].iloc[0]
    hist_dos = row[1:].to_numpy(dtype=float)

    json_path = os.path.join(JSON_FOLDER, f"{material}_persite.json")
    if not os.path.exists(json_path):
        print(f"JSON for {material} not found, skipping")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    if "tdos_per_site" not in data or SITE_KEY not in data["tdos_per_site"]:
        print(f"{material}: no site {SITE_KEY}")
        continue

    site_data = data["tdos_per_site"][SITE_KEY]
    raw_energies = np.array(site_data["energies"], dtype=float)
    raw_dos = np.array(site_data["densities"][SPIN], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].plot(raw_energies, raw_dos, color="blue", linewidth=1)
    axes[0].set_title("Raw DOS")
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("DOS")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].step(bin_centers, hist_dos, where="mid", color="orange", linewidth=1)
    axes[1].set_title("Histogrammed DOS")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"{material} | site {SITE_KEY}, spin {SPIN}", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = f"{material}_site{SITE_KEY}_spin{SPIN}_comparison.png"
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot: {out_path}")
