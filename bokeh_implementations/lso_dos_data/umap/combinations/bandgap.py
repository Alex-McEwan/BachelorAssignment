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

tdos_combo = [
    os.path.join(base_dir, "tdos", "tdos_spin1.csv"),    # tdos.up
    os.path.join(base_dir, "tdos", "tdos_spin-1.csv"),   # tdos.down
]

halides_paths = [
    os.path.join(base_dir, "halides", "spin1_sites5to10_summed.csv"),
    os.path.join(base_dir, "halides", "spin-1_sites5to10_summed.csv")
]

combo1_name = "b1up_b1down_b2up_b2down"
combo2_name = "tdosup_tdosdown_b1up_b1down_b2up_b2down"
combo3_name = "tdosup_tdosdown_b1up_b1down_b2up_b2down_xup_xdown"
combo4_name =  "b1up_b1down_b2up_b2down_xup_xdown"
halides_name = "Xup_Xdown"
tdos_combo_name = "tdosup_tdosdown"

# === Read and merge DOS data ===
dfs = []
for f in combo1:
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

bandgap_cols = list(bandgap_df.columns)
feature_columns = [c for c in merged.columns if c not in bandgap_cols + ["material"]]

X_sparse = sparse.csr_matrix(merged[feature_columns].values)

N_NEIGHBORS = 15
DISTANCE_METRIC = "manhattan"
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

plot_df = pd.DataFrame({
    "material": materials,
    "x": X_umap[:, 0],
    "y": X_umap[:, 1],
    "bandgap": bandgaps
})

report_base = "report"
tdos_base = "tdos"
DIRECTORY = "vacancy_ordered_umap_bandgap_color_fullrange"
SAVING_DIR = os.path.join("bokehfiles", report_base,tdos_base, DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"umap_bandgap_color_{combo1_name}_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df["bandgap"].min()),
    high=float(plot_df["bandgap"].max())
)

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"
plot = figure(
    width=900, height=900,
    tools=TOOLS,
    active_scroll="wheel_zoom"
)

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
    title="Bandgap (eV)",
    major_label_text_font_size="22pt",
    title_text_font_size="22pt"


)
plot.add_layout(color_bar, "right")

output_file(os.path.join(SAVING_DIR, FILE_NAME))
save(plot)

print(f"Bokeh plot saved to {os.path.join(SAVING_DIR, FILE_NAME)}")
