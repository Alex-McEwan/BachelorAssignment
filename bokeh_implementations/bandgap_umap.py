import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
import os

df = pd.read_csv("datasets/bandgap_data_v3.csv")

df = df.dropna()

print(df.head())

X = df.drop(columns=["Band gap values Clean", "Band gap units",  "chemicalFormula Clean", "index", "Reliability"])

X_scaled = StandardScaler().fit_transform(X)