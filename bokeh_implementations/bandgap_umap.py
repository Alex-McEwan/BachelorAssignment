import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Viridis256

from bokeh.palettes import Category10
import os

df = pd.read_csv("datasets/bandgap_data_v3.csv")

df = df.dropna()

print(df.head())

X = df.drop(columns=["Band gap values Clean", "Band gap units",  "chemicalFormula Clean", "index", "Reliability"])

X_scaled = StandardScaler().fit_transform(X)

y = df["Band gap values Clean"].to_numpy()

reducer = umap.UMAP(random_state=42)
print("started UMAP")
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

saving_dir = "bokehfiles"
os.makedirs(saving_dir, exist_ok=True)

X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
BANDGAP_STRING = "bandgap"
FORMULA_STRING = "formula"

plot_df = pd.DataFrame({
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    BANDGAP_STRING: y,
    FORMULA_STRING: df["chemicalFormula Clean"].to_numpy()

})

source = ColumnDataSource(plot_df)

color_mapping = LinearColorMapper(
    palette=Viridis256,
    low=float(plot_df[BANDGAP_STRING].min()),
    high=float(plot_df[BANDGAP_STRING].max())
)

plot = figure(
    title="Umap projection of the bandgap dataset",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source,
             color={"field": BANDGAP_STRING, "transform": color_mapping})

color_bar = ColorBar(color_mapper=color_mapping,
                     ticker=BasicTicker(),
                     label_standoff=8,
                     location=(0, 0),
                     title="Bandgap (eV)")

plot.add_layout(color_bar, "right")

plot.select_one(HoverTool).tooltips = [
    ("Bandgap (eV)", f"@{BANDGAP_STRING}{{0.000}}"),
    ("Horizontal axis", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Vertical axis", f"@{Y_AXIS_STRING}{{0.00}}"),
    ("Chemical Formula", f"@{FORMULA_STRING}")
]

output_file(os.path.join(saving_dir, "bandgap_umap.html"))
save(plot)




