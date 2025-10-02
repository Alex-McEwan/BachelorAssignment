import pandas as pd

df = pd.read_csv("datasets/output/dos_dataset_histogram_custom.csv")

# 1) Are there duplicate column names?
dups = df.columns[df.columns.duplicated()]
print("Duplicate columns:", list(dups))  # should be []

# 2) Do we actually have rows after dropping 'material'?
X = df.drop(columns=["material"])
print("Shape:", X.shape)  # (n_rows, n_features)

# 3) Are they numeric and do any rows have signal?
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
print("Rows with any nonzero:", (X != 0).any(axis=1).sum())
