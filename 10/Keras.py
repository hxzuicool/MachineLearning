# Python ≥3.5 is required
import sys
import pandas as pd

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras

# TensorFlow ≥2.0 is required
import tensorflow as tf

assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Ignore useless warnings (see SciPy issue #5998)
import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    print(X_test.shape)

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # n_rows = 4
    # n_cols = 10
    # plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    # for row in range(n_rows):
    #     for col in range(n_cols):
    #         index = n_cols * row + col
    #         plt.subplot(n_rows, n_cols, index + 1)
    #         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
    #         plt.axis('off')
    #         plt.title(class_names[y_train[index]], fontsize=12)
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # save_fig('fashion_mnist_plot', tight_layout=False)
    # plt.show()

    # model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # model.add(keras.layers.Dense(300, activation="relu"))
    # model.add(keras.layers.Dense(100, activation="relu"))
    # model.add(keras.layers.Dense(10, activation="softmax"))

    model = keras.models.load_model("keras_model.h5")

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    # history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)

    # pd.DataFrame(history.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # save_fig("keras_learning_curves_plot")
    # plt.show()

    # model.evaluate(X_test, y_test)

    # X_new = X_test[:3]
    # y_proba = model.predict(X_new)
    # y_proba.round(2)

    # model.save("keras_model.h5")