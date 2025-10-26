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

neighbours = 5
min_dist = 1.0
metric = 'euclidean'
random_state = 42


print("started UMAP")
reducer = umap.UMAP(n_neighbors=neighbours, min_dist=min_dist, metric=metric, random_state=random_state)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

saving_dir = os.path.join("bokehfiles", "report","digits_umap", "different_metrics")
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
plot.legend.label_text_font_size = "22pt"
plot.legend.title_text_font_size = "22pt"

file_name = f"digits_umap_{neighbours}_neighbors_{min_dist}_min_dist_{metric}_metric.html"
output_file(os.path.join(saving_dir, file_name))
print("saving plot to directory:", saving_dir, "as", file_name)
save(plot)

