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

base_dir = os.path.join("datasets", "output", "combinations_full_range")
combo1 = [
    os.path.join(base_dir, "BBAA", "site0_spin1.csv"),   # B1.up
    os.path.join(base_dir, "BBAA", "site1_spin1.csv"),   # B2.up
    os.path.join(base_dir, "BBAA", "site0_spin-1.csv"),  # B1.down
    os.path.join(base_dir, "BBAA", "site1_spin-1.csv")   # B2.down
]

combo2 = [
    os.path.join(base_dir, "tdos", "tdos_spin1.csv"),    # tdos.up
    os.path.join(base_dir, "tdos", "tdos_spin-1.csv"),   # tdos.down
    os.path.join(base_dir, "BBAA", "site0_spin1.csv"),   # B1.up
    os.path.join(base_dir, "BBAA", "site1_spin1.csv"),   # B2.up
    os.path.join(base_dir, "BBAA", "site0_spin-1.csv"),  # B1.down
    os.path.join(base_dir, "BBAA", "site1_spin-1.csv")   # B2.down
]

combo3 = [
    os.path.join(base_dir, "tdos", "tdos_spin1.csv"),    # tdos.up
    os.path.join(base_dir, "tdos", "tdos_spin-1.csv"),   # tdos.down
    os.path.join(base_dir, "BBAA", "site0_spin1.csv"),   # B1.up
    os.path.join(base_dir, "BBAA", "site1_spin1.csv"),   # B2.up
    os.path.join(base_dir, "BBAA", "site0_spin-1.csv"),  # B1.down
    os.path.join(base_dir, "BBAA", "site1_spin-1.csv"),  # B2.down
    os.path.join(base_dir, "halides", "spin1_sites5to10_summed.csv"),   # X.up
    os.path.join(base_dir, "halides", "spin-1_sites5to10_summed.csv")   # X.down
]

combo4 = [
    os.path.join(base_dir, "BBAA", "site0_spin1.csv"),   # B1.up
    os.path.join(base_dir, "BBAA", "site1_spin1.csv"),   # B2.up
    os.path.join(base_dir, "BBAA", "site0_spin-1.csv"),  # B1.down
    os.path.join(base_dir, "BBAA", "site1_spin-1.csv"),  # B2.down
    os.path.join(base_dir, "halides", "spin1_sites5to10_summed.csv"),   # X.up
    os.path.join(base_dir, "halides", "spin-1_sites5to10_summed.csv")   # X.down
]

halides_paths = [
    os.path.join(base_dir, "halides", "spin1_sites5to10_summed.csv"),
    os.path.join(base_dir, "halides", "spin-1_sites5to10_summed.csv")
]

combo1_name = "b1up_b1down_b2up_b2down"
combo2_name = "tdosup_tdosdown_b1up_b1down_b2up_b2down"
combo3_name = "tdosup_tdosdown_b1up_b1down_b2up_b2down_xup_xdown"
combo4_name =  "b1up_b1down_b2up_b2down_xup_xdown"

dfs = []
for f in combo2:
    df = pd.read_csv(f)
    prefix = os.path.splitext(os.path.basename(f))[0]
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "material"})
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="material", how="inner")

print(f"Shape of first file ({halides_paths[0]}): {dfs[0].shape}")

print(f"Merged dataset shape: {merged.shape}")
print(merged.head())

bandgap_csv_file = os.path.join("datasets", "output", "material_bandgap.csv")
bandgap_df = pd.read_csv(bandgap_csv_file)
merged = merged.merge(bandgap_df, on="material", how="inner")

materials = merged["material"].values
bandgaps = merged["bandgap"].values
feature_columns = [c for c in merged.columns if c not in ["material", "bandgap"]]
X_sparse = sparse.csr_matrix(merged[feature_columns].values)

N_NEIGHBORS = 15
DISTANCE_METRIC = "cosine"
DENSMAP = False

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

DIRECTORY = "combined_sparse_umap_bandgap_color_halide_marker_fullrange"
SAVING_DIR = os.path.join("bokehfiles", DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"combined_umap_halide_{combo2_name}_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
BANDGAP_STRING = "bandgap"
HALIDE_STRING = "halide"

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    BANDGAP_STRING: bandgaps,
    HALIDE_STRING: halides
})

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df[BANDGAP_STRING].min()),
    high=float(plot_df[BANDGAP_STRING].max())
)

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
    title=f"UMAP projection colored by bandgap & marker by halide ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
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
        color={"field": BANDGAP_STRING, "transform": color_mapping},
        legend_label=h
    )

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    ("Halide", f"@{HALIDE_STRING}"),
    ("Bandgap", f"@{BANDGAP_STRING}{{0.00}} eV"),
    (X_AXIS_STRING, f"@{X_AXIS_STRING}{{0.00}}"),
    (Y_AXIS_STRING, f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.xaxis.axis_label = X_AXIS_STRING
plot.yaxis.axis_label = Y_AXIS_STRING
plot.legend.title = "Halide"
plot.legend.location = "top_left"

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
