import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
from umap.parametric_umap import ParametricUMAP

import os

savingdir = r"bokeh_implementations/parametricumap/mnist_model"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))

encoder = Sequential([
    layers.InputLayer(input_shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(2)
])

embedder = ParametricUMAP(
    encoder=encoder,
    dims=(784,),
    verbose=True,
)

embedding = embedder.fit_transform(x_train)

embedder.save(os.path.join(savingdir))
