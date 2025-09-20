import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import umap

from scipy import sparse

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
import os

csv_file = os.path.join("datasets", "output", "dos_dataset_interpolated_5ev_around_fermi.csv")
df = pd.read_csv(csv_file)

materials = df["material"].values

energy_columns = [col for col in df.columns if col != "material"]

X_sparse = sparse.csr_matrix(df[energy_columns].values)

N_NEIGHBORS = 20

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)

print("started UMAP")
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

SAVING_DIR = os.path.join("bokehfiles")
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"dos_umap_sparse_5_ev_around_fermi_{N_NEIGHBORS}_neighbors.html"

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
})

source = ColumnDataSource(plot_df)

plot = figure(
    title=f"UMAP projection of DOS sparse dataset at interval around fermi energy with {N_NEIGHBORS} neighbors",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source, size=6, alpha=0.7)

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    (X_AXIS_STRING, f"@{X_AXIS_STRING}{{0.00}}"),
    (Y_AXIS_STRING, f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.xaxis.axis_label = X_AXIS_STRING
plot.yaxis.axis_label = Y_AXIS_STRING

output_file(os.path.join(SAVING_DIR, FILE_NAME))
save(plot)

print(f"Bokeh plot saved to {os.path.join(SAVING_DIR, FILE_NAME)}")
