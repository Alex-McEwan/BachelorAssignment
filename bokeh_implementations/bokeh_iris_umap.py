from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
import os

iris = load_iris(as_frame=True)
df = iris.frame
df = df.dropna()

X = df.drop(columns=["target"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df["target"].to_numpy()

print("started UMAP")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

species_names = iris.target_names

saving_dir = "bokehfiles"

SPECIES_STRING = "species"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"

species_strings = species_names[y]
plot_df = pd.DataFrame({
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    SPECIES_STRING: species_strings,
})

source = ColumnDataSource(plot_df)

color_mapping = CategoricalColorMapper(factors=list(species_names), palette=Category10[len(species_names)])

plot = figure(
    title="UMAP projection iris dataset",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter("x", "y", source=source,
          color={"field": SPECIES_STRING, "transform": color_mapping},
          legend_field=SPECIES_STRING)

plot.select_one(HoverTool).tooltips = [
    ("Species", f"@{SPECIES_STRING}"),
    ("Horizontal axis", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Vertical axis", f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.legend.title = "Species"
plot.legend.location = "top_left"
os.makedirs(saving_dir, exist_ok=True)
output_file(f"{saving_dir}/iris_umap_bokeh.html")
save(plot)
