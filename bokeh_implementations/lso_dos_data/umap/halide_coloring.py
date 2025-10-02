import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
from scipy import sparse
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
import os
import re

csv_file = os.path.join("datasets", "output", "dos_dataset_histogram_5_ev_cutoff_after_bandgap.csv")
bandgap_csv_file = os.path.join("datasets", "output", "material_bandgap.csv")

dos_df = pd.read_csv(csv_file)
bandgap_df = pd.read_csv(bandgap_csv_file)
df = dos_df.merge(bandgap_df, on="material", how="inner")

materials = df["material"].values
energy_columns = [col for col in df.columns if col not in ["material", "bandgap"]]
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

DIRECTORY = "dos_sparse_histogram_halide_coloring"
SAVING_DIR = os.path.join("bokehfiles", DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"dos_sparse_histogram_umap_halide_5_ev_after_cbm_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
HALIDE_STRING = "halide"

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    HALIDE_STRING: halides
})

source = ColumnDataSource(plot_df)

unique_halides = sorted(set(halides))
palette = Category10[max(3, min(len(unique_halides), 10))]
color_mapping = CategoricalColorMapper(factors=unique_halides, palette=palette)

plot = figure(
    title=f"UMAP projection of DOS dataset colored by halide ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source, size=6, alpha=0.7,
             color={"field": HALIDE_STRING, "transform": color_mapping},
             legend_field=HALIDE_STRING)

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    ("Halide", f"@{HALIDE_STRING}"),
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
