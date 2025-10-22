import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
from scipy import sparse
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Viridis256
import os
import re

# === Paths ===
base_dir = os.path.join("datasets", "output", "combinations_full_range", "vacancy_ordered")
tdos_combo = [
    os.path.join(base_dir, "tdos", "tdos_spin1.csv"),    # tdos.up
    os.path.join(base_dir, "tdos", "tdos_spin-1.csv"),   # tdos.down
]

# === Read and merge DOS data ===
dfs = []
for f in tdos_combo:
    df = pd.read_csv(f)
    prefix = os.path.splitext(os.path.basename(f))[0]
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "material"})
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="material", how="inner")

print(f"Merged dataset shape: {merged.shape}")

# === Add bandgap data ===
bandgap_csv_file = os.path.join("datasets", "output", "material_bandgap.csv")
bandgap_df = pd.read_csv(bandgap_csv_file)
merged = merged.merge(bandgap_df, on="material", how="inner")

materials = merged["material"].values
bandgaps = merged["bandgap"].values

# Remove any non-feature columns (bandgap info, material)
bandgap_cols = list(bandgap_df.columns)
feature_columns = [c for c in merged.columns if c not in bandgap_cols + ["material"]]

X_sparse = sparse.csr_matrix(merged[feature_columns].values)

# === Scale and run UMAP ===
N_NEIGHBORS = 15
DISTANCE_METRIC = "euclidean"
DENSMAP = False

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)

print("started UMAP")
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    metric=DISTANCE_METRIC,
    random_state=42,
    densmap=DENSMAP
)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

# === Prepare data for plotting ===
plot_df = pd.DataFrame({
    "material": materials,
    "x": X_umap[:, 0],
    "y": X_umap[:, 1],
    "bandgap": bandgaps
})

# === Bokeh setup ===
report_base = "report" 
DIRECTORY = "vacancy_ordered_umap_bandgap_color_fullrange"
SAVING_DIR = os.path.join("bokehfiles", report_base, DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"umap_bandgap_color_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df["bandgap"].min()),
    high=float(plot_df["bandgap"].max())
)

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"
plot = figure(
    title=f"UMAP projection colored by bandgap ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
    width=900, height=900,
    tools=TOOLS,
    active_scroll="wheel_zoom"
)

# === Scatter (single marker, colored by bandgap) ===
source = ColumnDataSource(plot_df)
plot.circle(
    "x", "y",
    source=source,
    size=7, alpha=0.7,
    color={"field": "bandgap", "transform": color_mapping}
)

# === Tooltip setup ===
plot.select_one(HoverTool).tooltips = [
    ("Material", "@material"),
    ("Bandgap", "@bandgap{0.00} eV"),
    ("x", "@x{0.00}"),
    ("y", "@y{0.00}")
]

# === Axes and color bar ===
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

output_file(os.path.join(SAVING_DIR, FILE_NAME))
save(plot)

print(f"Bokeh plot saved to {os.path.join(SAVING_DIR, FILE_NAME)}")
