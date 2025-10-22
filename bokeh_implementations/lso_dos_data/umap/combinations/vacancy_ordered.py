import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import umap
from scipy import sparse
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
import os

base_dir = os.path.join("datasets", "output", "combinations_full_range", "vacancy_ordered")

halides_paths = [
    os.path.join(base_dir, "halides", "spin1_sites5to10_summed.csv"),
    os.path.join(base_dir, "halides", "spin-1_sites5to10_summed.csv")
]

dfs = []
for f in halides_paths:
    df = pd.read_csv(f)
    print(f"Number of columns in {f}: {df.shape[1]}")
    prefix = os.path.splitext(os.path.basename(f))[0]
    df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "material"})
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="material", how="inner")

print(f"Number of rows in merged dataset: {merged.shape[0]}")
print(f"Merged dataset shape: {merged.shape}")

vacancy_csv_file = os.path.join("datasets", "output", "vacancy_ordered_labels.csv")
vacancy_df = pd.read_csv(vacancy_csv_file)
merged = merged.merge(vacancy_df, on="material", how="inner")

materials = merged["material"].values
vacancy_flags = merged["vacancy_ordered"].values

feature_columns = [c for c in merged.columns if c not in ["material", "vacancy_ordered"]]
X_sparse = sparse.csr_matrix(merged[feature_columns].values)

N_NEIGHBORS = 15
DISTANCE_METRIC = "euclidean"
DENSMAP = False

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)

print("started UMAP")
reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, metric=DISTANCE_METRIC, random_state=42, densmap=DENSMAP)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

report_base = "report"
DIRECTORY = "vacancy_ordered_combined_sparse_umap_vacancy_coloring_fullrange"
SAVING_DIR = os.path.join("bokehfiles", report_base, DIRECTORY)
os.makedirs(SAVING_DIR, exist_ok=True)

FILE_NAME = f"combined_umap_vacancycolor_{N_NEIGHBORS}_neighbors_{DISTANCE_METRIC}_densmap_{DENSMAP}.html"

MATERIAL_STRING = "material"
X_AXIS_STRING = "x"
Y_AXIS_STRING = "y"
VACANCY_STRING = "vacancy_ordered"

plot_df = pd.DataFrame({
    MATERIAL_STRING: materials,
    X_AXIS_STRING: X_umap[:, 0],
    Y_AXIS_STRING: X_umap[:, 1],
    VACANCY_STRING: vacancy_flags
})

plot_df[VACANCY_STRING] = plot_df[VACANCY_STRING].map({True: "Vacancy-ordered", False: "Normal"})

source = ColumnDataSource(plot_df)

unique_states = sorted(set(plot_df[VACANCY_STRING]))
palette = Category10[max(3, min(len(unique_states), 10))]
color_mapping = CategoricalColorMapper(factors=unique_states, palette=palette)

plot = figure(
    title=f"UMAP projection of combined dataset colored by vacancy ordering ({N_NEIGHBORS} neighbors, {DISTANCE_METRIC} metric)",
    width=800, height=800,
    tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    active_scroll="wheel_zoom"
)

plot.scatter(X_AXIS_STRING, Y_AXIS_STRING, source=source, size=6, alpha=0.7,
             color={"field": VACANCY_STRING, "transform": color_mapping},
             legend_field=VACANCY_STRING)

plot.select_one(HoverTool).tooltips = [
    ("Material", f"@{MATERIAL_STRING}"),
    ("Vacancy Ordered", f"@{VACANCY_STRING}"),
    (X_AXIS_STRING, f"@{X_AXIS_STRING}{{0.00}}"),
    (Y_AXIS_STRING, f"@{Y_AXIS_STRING}{{0.00}}"),
]

plot.xaxis.axis_label = X_AXIS_STRING
plot.yaxis.axis_label = Y_AXIS_STRING
plot.legend.title = "Vacancy Ordered"
plot.legend.location = "top_left"

output_file(os.path.join(SAVING_DIR, FILE_NAME))
save(plot)

print(f"Bokeh plot saved to {os.path.join(SAVING_DIR, FILE_NAME)}")
