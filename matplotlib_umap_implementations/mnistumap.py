
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap 
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits(as_frame=True)
df = digits.frame
print(df.head())


X = df.drop(columns=["target"]) 
y = df["target"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.shape)


reducer = umap.UMAP()
X_umap = reducer.fit_transform(X_scaled)
print(X_umap.shape)

palette = sns.color_palette("tab10", 10)  

for digit in range(10):
    plt.scatter(
        X_umap[y == digit, 0],
        X_umap[y == digit, 1],
        label=str(digit),
        c=[palette[digit]],
        s=5
    )

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP projection of the Digits dataset")
plt.legend(title="Digit", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("plots/digits_umap.png", dpi=300, bbox_inches="tight")
