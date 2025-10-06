import os
import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------- USER INPUT -------------
CSV_TO_CHECK = os.path.join("datasets", "output", "per_site", "site0_spin-1.csv")
SITE_KEY = "0"    # set to match the CSV you picked
SPIN = "-1"       # set to match the CSV you picked
JSON_FOLDER = os.path.join("datasets", "lsodos_persitejsons_250930")
PLOT_DIR = os.path.join("datasets", "output", "plots", "check_csv")
os.makedirs(PLOT_DIR, exist_ok=True)
RANDOM_SEED = 42
N_SAMPLES = 5
# --------------------------------------

random.seed(RANDOM_SEED)

# Load the generated histogram CSV
df_hist = pd.read_csv(CSV_TO_CHECK)
materials = df_hist["material"].tolist()
energy_cols = df_hist.columns[1:]
bin_centers = [float(col.split("=")[1].replace("eV", "")) for col in energy_cols]

# Pick random subset
sample_materials = random.sample(materials, min(N_SAMPLES, len(materials)))

for material in sample_materials:
    # --- Histogrammed data ---
    row = df_hist[df_hist["material"] == material].iloc[0]
    hist_dos = row[1:].to_numpy(dtype=float)

    # --- Original raw data from JSON ---
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

    # --- Plot ---
    plt.figure(figsize=(10,5))
    plt.plot(raw_energies, raw_dos, label="Raw", color="blue", linewidth=1)
    plt.step(bin_centers, hist_dos, where="mid", label="Histogrammed", color="orange", linewidth=1)
    plt.title(f"{material} | site {SITE_KEY}, spin {SPIN}")
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_name = f"{material}_site{SITE_KEY}_spin{SPIN}_comparison.png"
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot: {out_path}")
