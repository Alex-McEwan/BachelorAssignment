import pandas as pd
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource
from bokeh.io import save

output_folder = "bokehfiles/"

df = pd.DataFrame({
    "x": list(range(0, 11)), 
    "y": list(range(0, 11))    
})

source = ColumnDataSource(df)

p = figure(title="y = x line", x_axis_label="x", y_axis_label="y")

p.circle("x", "y", source=source, size=8, color="blue", alpha=0.5)

output_file(output_folder + "bokehplot.html")

save(p)
