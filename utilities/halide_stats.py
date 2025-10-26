import pandas as pd
import os
from scipy import stats

# === Load data ===
path = os.path.join("datasets", "LSODOS", "data_luc", "CombinedHDPinfo_lsodos.csv")
df = pd.read_csv(path)

# === Keep only needed columns ===
subset = df[["comp", "CBM", "VBM", "bandgap"]].copy()

# === Identify halide ===
HALIDES = ["Cl", "Br", "I", "F", "At", "Ts"]

def extract_halide(name: str) -> str:
    for h in HALIDES:
        if h in name:
            return h
    print(f"⚠️  WARNING: No halide found in material name: {name}")
    return "UNKNOWN"

subset["halide"] = subset["comp"].apply(extract_halide)

# === Group by halide and compute stats ===
group_stats = subset.groupby("halide").agg(
    count=("halide", "count"),
    CBM_mean=("CBM", "mean"),
    CBM_std=("CBM", "std"),
    CBM_var=("CBM", "var"),
    VBM_mean=("VBM", "mean"),
    VBM_std=("VBM", "std"),
    VBM_var=("VBM", "var"),
    bandgap_mean=("bandgap", "mean"),
    bandgap_std=("bandgap", "std"),
    bandgap_var=("bandgap", "var")
)

print("\n=== CBM / VBM / Bandgap statistics per halide ===")
print(group_stats.round(4))

# === ANOVA tests (assume halides present, warn if unknown) ===
halides_present = [h for h in group_stats.index if h != "UNKNOWN"]
if not halides_present:
    print("\n❌ No valid halides found in dataset — check formula names.")
else:
    cbm_groups = [subset.loc[subset["halide"] == h, "CBM"].dropna() for h in halides_present]
    vbm_groups = [subset.loc[subset["halide"] == h, "VBM"].dropna() for h in halides_present]
    gap_groups = [subset.loc[subset["halide"] == h, "bandgap"].dropna() for h in halides_present]

    f_cbm, p_cbm = stats.f_oneway(*cbm_groups)
    f_vbm, p_vbm = stats.f_oneway(*vbm_groups)
    f_gap, p_gap = stats.f_oneway(*gap_groups)

    print("\n=== ANOVA significance ===")
    print(f"CBM differences between halides:     F={f_cbm:.3f}, p={p_cbm:.3e}")
    print(f"VBM differences between halides:     F={f_vbm:.3f}, p={p_vbm:.3e}")
    print(f"Bandgap differences between halides: F={f_gap:.3f}, p={p_gap:.3e}")
