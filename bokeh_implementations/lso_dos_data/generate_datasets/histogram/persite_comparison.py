import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
OUTPUT_DIR = os.path.join("datasets", "output")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_per_channel")
HIST_DIR = os.path.join(OUTPUT_DIR, "one_site_histograms")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots", "comparison_persite_hist")
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_SEED = 42  # fixed seed for reproducible selection
# --------------

random.seed(RANDOM_SEED)

# Collect all pairs that exist in both dirs
pairs = []
for raw_file in os.listdir(RAW_DIR):
    if raw_file.endswith("_raw.csv"):
        hist_file = raw_file.replace("_raw.csv", "_hist.csv")
        if os.path.exists(os.path.join(HIST_DIR, hist_file)):
            pairs.append((raw_file, hist_file))

if not pairs:
    print("No matching raw/hist pairs found!")
    exit()

# Pick 5 random pairs (or less if fewer available)
sample_pairs = random.sample(pairs, min(5, len(pairs)))

for raw_file, hist_file in sample_pairs:
    raw_path = os.path.join(RAW_DIR, raw_file)
    hist_path = os.path.join(HIST_DIR, hist_file)

    raw_df = pd.read_csv(raw_path)
    hist_df = pd.read_csv(hist_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left plot: raw DOS
    axes[0].plot(raw_df["Energy(eV)"], raw_df["DOS"], color="blue", linewidth=1)
    axes[0].set_title("Raw DOS")
    axes[0].set_xlabel("Energy (eV)")
    axes[0].set_ylabel("DOS")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right plot: histogrammed DOS
    axes[1].step(hist_df["Energy_center(eV)"], hist_df["DOS_binned"],
                 where="mid", color="orange", linewidth=1)
    axes[1].set_title("Histogrammed DOS")
    axes[1].set_xlabel("Energy (eV)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(raw_file.replace("_raw.csv", ""), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = raw_file.replace("_raw.csv", "_comparison_side_by_side.png")
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot: {out_path}")
