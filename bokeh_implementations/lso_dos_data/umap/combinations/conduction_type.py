import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
from scipy import sparse
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Inferno256, Viridis256, Cividis256
import os

base_dir = os.path.join("datasets", "output", "combinations_full_range", "vacancy_ordered")

tdos_combo = [
    os.path.join(base_dir, "tdos", "tdos_spin1.csv"),
    os.path.join(base_dir, "tdos", "tdos_spin-1.csv"),
]

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

bandgap_csv_file = os.path.join("datasets", "output", "material_bandgap.csv")
bandgap_df = pd.read_csv(bandgap_csv_file)
merged = merged.merge(bandgap_df, on="material", how="inner")

materials = merged["material"].values
bandgaps = merged["bandgap"].values
conduction_types = merged["cond_type"].values

bandgap_cols = list(bandgap_df.columns)
feature_columns = [c for c in merged.columns if c not in bandgap_cols + ["material"]]
X_sparse = sparse.csr_matrix(merged[feature_columns].values)

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

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
BANDGAP_STRING = "bandgap"
COND_TYPE_STRING = "cond_type"

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    BANDGAP_STRING: bandgaps,
    COND_TYPE_STRING: conduction_types
})

report_base = "report"
DIRECTORY = "vacancy_ordered_umap_bandgap_color_condtype_marker_fullrange"
SAVING_DIR = os.path.join("bokehfiles", report_base, DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"umap_bandgap_color_condtype_marker_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

color_mapping = LinearColorMapper(
    palette=Cividis256,
    low=float(plot_df[BANDGAP_STRING].quantile(0.01)),
    high=float(plot_df[BANDGAP_STRING].quantile(0.99))
)

marker_map = {
    "insulator": "square",
    "semiconductor": "circle",
    "half-metal": "triangle",
    "metallic": "diamond"
}

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"
plot = figure(
    title=f"UMAP projection colored by bandgap & marker by conduction type ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
    width=900, height=900,
    tools=TOOLS,
    active_scroll="wheel_zoom"
)

unique_cond_types = sorted(set(conduction_types))

for ctype in unique_cond_types:
    sub = plot_df[plot_df[COND_TYPE_STRING] == ctype]
    source = ColumnDataSource(sub)
    marker = marker_map.get(ctype.lower(), "circle")
    getattr(plot, marker)(
        X_AXIS_STRING, Y_AXIS_STRING,
        source=source, size=7, alpha=0.7,
        color={"field": BANDGAP_STRING, "transform": color_mapping},
        legend_label=ctype
    )

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    ("Conduction Type", f"@{COND_TYPE_STRING}"),
    ("Bandgap", f"@{BANDGAP_STRING}{{0.00}} eV"),
    (X_AXIS_STRING, f"@{X_AXIS_STRING}{{0.00}}"),
    (Y_AXIS_STRING, f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.xaxis.axis_label = X_AXIS_STRING
plot.yaxis.axis_label = Y_AXIS_STRING
plot.legend.title = "Conduction Type"
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
