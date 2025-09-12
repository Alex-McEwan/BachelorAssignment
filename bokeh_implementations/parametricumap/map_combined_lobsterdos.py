import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
import os

df = pd.read_csv(r"datasets\data_luc\CombinedHDPinfo_lobsterdos.csv")

numeric_cols = [
    "VBM", "CBM", "bandgap",
    "spin_vbm", "spin_cbm",
    "magmom_tot_vasp", "magmom_tot_lobster", "popdiff_tot_lobster"
]

X = df[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("started UMAP")
reducer = umap.UMAP()
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

plot = figure(
    title="UMAP projection of Combined LOBSTER data",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source,
             color="navy", alpha=0.6, size=6)

plot.select_one(HoverTool).tooltips = [
    ("Compound", f"@{COMP_STRING}"),
    ("Bandgap", f"@{BANDGAP_STRING}{{0.00}} eV"),
    ("X", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Y", f"@{Y_AXIS_STRING}{{0.00}}"),
]

output_file(os.path.join(saving_dir, "lobster_umap.html"))
save(plot)
