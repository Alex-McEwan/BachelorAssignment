import os
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
OUTPUT_DIR = os.path.join("datasets", "output")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw_per_channel")
HIST_DIR = os.path.join(OUTPUT_DIR, "one_site_histograms")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots", "comparison_persite_hist")
os.makedirs(PLOT_DIR, exist_ok=True)
# --------------

# Get all raw files
raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith("_raw.csv")]

for raw_file in raw_files:
    # Build matching hist filename
    hist_file = raw_file.replace("_raw.csv", "_hist.csv")
    hist_path = os.path.join(HIST_DIR, hist_file)

    # Skip if no matching histogram
    if not os.path.exists(hist_path):
        print(f"Skipping {raw_file}: no matching histogram found")
        continue

    raw_path = os.path.join(RAW_DIR, raw_file)
    raw_df = pd.read_csv(raw_path)
    hist_df = pd.read_csv(hist_path)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(raw_df["Energy(eV)"], raw_df["DOS"], label="Raw", color="blue", linewidth=1)
    plt.step(hist_df["Energy_center(eV)"], hist_df["DOS_binned"], where="mid",
             label="Histogram", color="orange", linewidth=1)

    plt.title(raw_file.replace("_raw.csv", ""), fontsize=10)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.legend()
    plt.tight_layout()

    # Save plot
    out_name = raw_file.replace("_raw.csv", "_comparison.png")
    out_path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot: {out_path}")
