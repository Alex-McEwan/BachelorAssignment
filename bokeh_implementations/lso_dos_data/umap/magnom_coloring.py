import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
from scipy import sparse
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
import os
import re

base_csv = os.path.join("datasets", "output", "dos_dataset_histogram_5_ev_cutoff_after_bandgap.csv")
magmom_csv = os.path.join("datasets", "output", "material_bandgap.csv")  # file with magmom_tot_lobster column

dos_df = pd.read_csv(base_csv)
magmom_df = pd.read_csv(magmom_csv)
df = dos_df.merge(magmom_df, on="material", how="inner")

materials = df["material"].values
magmoms = df["magmom_tot_lobster"].values
energy_columns = [c for c in df.columns if c not in ["material", "magmom_tot_lobster"]]

X_sparse = sparse.csr_matrix(df[energy_columns].values)

N_NEIGHBORS = 15
DISTANCE_METRIC = "cosine"
DENSMAP = True

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)

print("started UMAP")
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, metric=DISTANCE_METRIC, random_state=42, densmap=DENSMAP)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

HALIDES = ['Cl', 'Br', 'I', 'F', 'At', 'Ts']
def extract_halide(name: str) -> str:
    m = re.search(r'_(.*?)_lsodos$', name)
    formula = m.group(1) if m else name
    for h in HALIDES:
        if formula.endswith(h):
            return h
    return "Unknown"

halides = [extract_halide(m) for m in materials]
unique_halides = sorted(set(halides))

DIRECTORY = "dos_umap_magmom_color_halide_marker"
SAVING_DIR = os.path.join("bokehfiles", DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"dos_umap_magmom_color_halide_marker_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
MAGMOM_STRING = "magmom"
HALIDE_STRING = "halide"

# Assign colors based on sign of magmom
colors = np.where(magmoms > 0, "orange", "blue")

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    MAGMOM_STRING: magmoms,
    HALIDE_STRING: halides,
    "color": colors
})

marker_map = {
    "Cl": "circle",
    "Br": "triangle",
    "I": "square",
    "F": "diamond",
    "At": "cross",
    "Ts": "asterisk",
    "Unknown": "circle_x"
}

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"
plot = figure(
    title=f"UMAP projection colored by magnetic moment sign & marker by halide ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
    width=900, height=900,
    tools=TOOLS,
    active_scroll="wheel_zoom"
)

for h in unique_halides:
    sub = plot_df[plot_df[HALIDE_STRING] == h]
    source = ColumnDataSource(sub)
    marker = marker_map.get(h, "circle")
    getattr(plot, marker)(
        X_AXIS_STRING, Y_AXIS_STRING,
        source=source, size=7, alpha=0.7,
        color="color",
        legend_label=h
    )

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    ("Halide", f"@{HALIDE_STRING}"),
    ("Magmom", f"@{MAGMOM_STRING}{{0.00}}"),
    (X_AXIS_STRING, f"@{X_AXIS_STRING}{{0.00}}"),
    (Y_AXIS_STRING, f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.xaxis.axis_label = X_AXIS_STRING
plot.yaxis.axis_label = Y_AXIS_STRING
plot.legend.title = "Halide"
plot.legend.location = "top_left"

output_file(os.path.join(SAVING_DIR, FILE_NAME))
save(plot)

print(f"Bokeh plot saved to {os.path.join(SAVING_DIR, FILE_NAME)}")
