# umap_runner.py
import os
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
import umap

class UMAPReducer:
    def __init__(self, csv_files, n_neighbors=15, metric="cosine", densmap=True):
        self.csv_files = csv_files if isinstance(csv_files, list) else [csv_files]
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.densmap = densmap

    def load_and_merge(self) -> pd.DataFrame:
        merged = None
        for f in self.csv_files:
            df = pd.read_csv(f)
            prefix = os.path.splitext(os.path.basename(f))[0]
            df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "material"})
            merged = df if merged is None else merged.merge(df, on="material", how="inner")
        return merged

    def fit_transform(self, df: pd.DataFrame):
        features = [c for c in df.columns if c != "material"]
        X = sparse.csr_matrix(df[features].values)
        X_scaled = MaxAbsScaler().fit_transform(X)
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            random_state=42,
            densmap=self.densmap
        )
        return reducer.fit_transform(X_scaled)

    def run(self):
        df = self.load_and_merge()
        embedding = self.fit_transform(df)
        return df, embedding
