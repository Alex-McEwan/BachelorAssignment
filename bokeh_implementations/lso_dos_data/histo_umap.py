import os
import json
import numpy as np
import pandas as pd

from pymatgen.electronic_structure.dos import LobsterCompleteDos

from sklearn.preprocessing import MaxAbsScaler
from scipy import sparse
import umap

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Viridis256



# ----------------------------
# SETTINGS
# ----------------------------
INPUT_FOLDER = os.path.join("datasets", "completedosjsons_lsodos")
BANDGAP_FILE = os.path.join("datasets", "output", "material_bandgap.csv")

# fingerprint settings
MIN_E = -5.0   # energy window below Ef
MAX_E =  5.0   # energy window above Ef
N_BINS = 256   # histogram bins
FP_TYPE = "summed_pdos"  # or "tdos", "s_pdos", etc.

# UMAP settings
N_NEIGHBORS = 15
DISTANCE_METRIC = "euclidean"

# Output settings
OUTPUT_DIR = os.path.join("datasets", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUTPUT_DIR, f"dos_fingerprints_{FP_TYPE}_{N_BINS}bins.csv")

BOKEH_DIR = "bokehfiles"
os.makedirs(BOKEH_DIR, exist_ok=True)
BOKEH_FILE = os.path.join(
    BOKEH_DIR,
    f"dos_umap_{FP_TYPE}_{N_BINS}bins_{N_NEIGHBORS}neighbors_{DISTANCE_METRIC}.html"
)


# ----------------------------
# STEP 1: Build dataset
# ----------------------------
rows = []
for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(INPUT_FOLDER, fname)
    with open(fpath, "r") as f:
        data = json.load(f)

    try:
        dos = LobsterCompleteDos.from_dict(data)
        fp = dos.get_dos_fp(
            fp_type=FP_TYPE,
            min_e=MIN_E,
            max_e=MAX_E,
            n_bins=N_BINS,
            normalize=True
        )
    except Exception as e:
        print(f"Skipping {fname}, error: {e}")
        continue

    material = os.path.splitext(fname)[0]
    rows.append([material, *fp.densities])

    print(f"Processed {material}, fp length={len(fp.densities)}")

# make dataframe
colnames = ["material"] + [f"bin{i}" for i in range(N_BINS)]
df = pd.DataFrame(rows, columns=colnames)

# save CSV
df.to_csv(CSV_OUT, index=False)
print(f"Saved fingerprints to {CSV_OUT}")


# ----------------------------
# STEP 2: Merge with bandgaps
# ----------------------------
bandgap_df = pd.read_csv(BANDGAP_FILE)
df = df.merge(bandgap_df, on="material", how="inner")

materials = df["material"].values
bandgaps = df["bandgap"].values

X = sparse.csr_matrix(df.drop(columns=["material", "bandgap"]).values)
X_scaled = MaxAbsScaler().fit_transform(X)


# ----------------------------
# STEP 3: Run UMAP
# ----------------------------
print("Started UMAP...")
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, metric=DISTANCE_METRIC, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("Finished UMAP.")


# ----------------------------
# STEP 4: Make Bokeh plot
# ----------------------------
plot_df = pd.DataFrame({
    "material": materials,
    "x": X_umap[:, 0],
    "y": X_umap[:, 1],
    "bandgap": bandgaps
})

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df["bandgap"].min()),
    high=float(plot_df["bandgap"].max())
)

source = ColumnDataSource(plot_df)

plot = figure(
    title=f"UMAP projection of DOS fingerprints ({FP_TYPE}, {N_BINS} bins, {N_NEIGHBORS} neighbors, {DISTANCE_METRIC})",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter("x", "y", source=source, size=6, alpha=0.7,
             color={"field": "bandgap", "transform": color_mapping})

plot.select_one(HoverTool).tooltips = [
    ("Material", "@material"),
    ("x", "@x{0.00}"),
    ("y", "@y{0.00}"),
    ("Bandgap", "@bandgap{0.00} eV"),
]

plot.xaxis.axis_label = "x"
plot.yaxis.axis_label = "y"

color_bar = ColorBar(
    color_mapper=color_mapping,
    ticker=BasicTicker(),
    label_standoff=8,
    location=(0, 0),
    title="Bandgap (eV)"
)
plot.add_layout(color_bar, "right")

output_file(BOKEH_FILE)
save(plot)
print(f"Bokeh plot saved to {BOKEH_FILE}")
