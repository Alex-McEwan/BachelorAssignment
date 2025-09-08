import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource

df = pd.DataFrame({
    "x": list(range(0, 11)), 
    "y": list(range(0, 11))    
})

source = ColumnDataSource(df)

p = figure(title="y = x line", x_axis_label="x", y_axis_label="y")

p.line("x", "y", source=source, line_width=2)

output_file("bokehpractise.html")
show(p)
