import os
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from umap.parametric_umap import load_ParametricUMAP

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10

model_path = r"bokeh_implementations/parametricumap/mnist_model"
embedder = load_ParametricUMAP(model_path)

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape((x_test.shape[0], -1))

X_umap = embedder.transform(x_test)

DIGIT_STRING = "digit"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"

digit_strings = y_test.astype(str)
plot_df = pd.DataFrame({
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    DIGIT_STRING: digit_strings,
})

source = ColumnDataSource(plot_df)
color_mapping = CategoricalColorMapper(
    factors=[str(digit) for digit in range(10)],
    palette=Category10[10]
)

plot = figure(
    title="Parametric UMAP projection of MNIST test set",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(
    X_AXIS_STRING, Y_AXIS_STRING,
    source=source,
    color={"field": DIGIT_STRING, "transform": color_mapping},
    legend_field=DIGIT_STRING,
    size=4
)

plot.select_one(HoverTool).tooltips = [
    ("Digit", f"@{DIGIT_STRING}"),
    ("X", f"@{X_AXIS_STRING}{{0.00}}"),
    ("Y", f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.legend.title = "Digit"
plot.legend.location = "top_left"

saving_dir = "bokehfiles"
os.makedirs(saving_dir, exist_ok=True)
output_file(os.path.join(saving_dir, "mnist_parametric_umap.html"))
save(plot)
