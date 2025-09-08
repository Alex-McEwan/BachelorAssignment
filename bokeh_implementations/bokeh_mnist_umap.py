from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
import os

digits = load_digits(as_frame=True)
df = digits.frame

X = df.drop(columns=["target"]) 
y = df["target"].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("started UMAP")
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

saving_dir = "bokehfiles"
os.makedirs(saving_dir, exist_ok=True)

DIGIT_STRING = "digit"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"

digits_names = [str(d) for d in digits.target_names]
digit_strings = y.astype(str) 
plot_df = pd.DataFrame({
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    DIGIT_STRING: digit_strings,
})

source = ColumnDataSource(plot_df)

color_mapping = CategoricalColorMapper(factors=list(digits_names), palette=Category10[len(digits_names)])

plot = figure(
    title="UMAP projection digits dataset",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter("x", "y", source=source,
            color={"field": DIGIT_STRING, "transform": color_mapping},
            legend_field=DIGIT_STRING)

plot.select_one(HoverTool).tooltips = [
    ("Digit", f"@{DIGIT_STRING}"),
    ("Horizontal axis", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Vertical axis", f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.legend.title = "Digit"
plot.legend.location = "top_left"

output_file(os.path.join(saving_dir, "digits_umap.html"))
save(plot)

