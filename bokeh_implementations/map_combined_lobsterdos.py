import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Viridis256
import os

df = pd.read_csv(os.path.join("datasets", "data_luc", "CombinedHDPinfo_lobsterdos.csv"))

numeric_cols = [
    "VBM", "CBM", "bandgap",
    "spin_vbm", "spin_cbm",
    "magmom_tot_vasp", "magmom_tot_lobster", "popdiff_tot_lobster"
]

X = df[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("started UMAP")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

saving_dir = "bokehfiles"
os.makedirs(saving_dir, exist_ok=True)

X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
COMP_STRING = "compound"
BANDGAP_STRING = "bandgap"

plot_df = pd.DataFrame({
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    COMP_STRING: df["comp"],
    BANDGAP_STRING: df["bandgap"],
})

source = ColumnDataSource(plot_df)

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df[BANDGAP_STRING].min()),
    high=float(plot_df[BANDGAP_STRING].max())
)

plot = figure(
    title="UMAP projection of Combined lobsterdos data colored by bandgap",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source,
             color={"field": BANDGAP_STRING, "transform": color_mapping},
             size=6, alpha=0.7)

color_bar = ColorBar(color_mapper=color_mapping,
                     ticker=BasicTicker(),
                     label_standoff=8,
                     location=(0, 0),
                     title="Bandgap (eV)")

plot.add_layout(color_bar, "right")

plot.select_one(HoverTool).tooltips = [
    ("Compound", f"@{COMP_STRING}"),
    ("Bandgap (eV)", f"@{BANDGAP_STRING}{{0.000}}"),
    ("X", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Y", f"@{Y_AXIS_STRING}{{0.00}}"),
]

output_file(os.path.join(saving_dir, "lobsterdos_umap_bandgap.html"))
save(plot)
