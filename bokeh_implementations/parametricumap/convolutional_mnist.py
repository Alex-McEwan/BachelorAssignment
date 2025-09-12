import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
from umap.parametric_umap import ParametricUMAP
import os

savingdir = r"bokeh_implementations/parametricumap/convolutional_model"
os.makedirs(savingdir, exist_ok=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = x_train[:1000], y_train[:1000]

x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  

encoder = Sequential([
    layers.Input(shape=(28, 28, 1)),  
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(2)
])

x_train_flat = x_train.reshape((x_train.shape[0], -1)) 

embedder = ParametricUMAP(
    encoder=encoder,
    dims=(28, 28, 1),  
    verbose=True,
    keras_fit_kwargs={"epochs": 1, "batch_size": 128}
)

embedding = embedder.fit_transform(x_train_flat)
