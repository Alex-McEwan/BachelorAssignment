from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris(as_frame=True)
df = iris.frame
df = df.dropna()

X = df.drop(columns=["target"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = df["target"].to_numpy()

print("started UMAP")
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
print("finished UMAP")

species_names = iris.target_names

palette = sns.color_palette("tab10", len(species_names))

for i, name in enumerate(species_names):
    plt.scatter(
        X_umap[y == i, 0],
        X_umap[y == i, 1],
        label=name,
        s=5,
        c=[palette[i]]
    )

plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP projection of Iris dataset")
plt.savefig("plots/iris_umap.png", dpi=300, bbox_inches="tight")
