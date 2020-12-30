import sys
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
assert tf.__version__ >= "2.0"
assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)
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


warnings.filterwarnings(action="ignore", message="^internal gelsd")

if __name__ == '__main__':
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=['accuracy'])
    # history = model.fit(X_train, y_train, epochs=40, validation_data=(X_valid, y_valid))
    history = model.fit(X_train, y_train, epochs=40, validation_split=0.1)
    mse_test = model.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_pred = model.predict(X_new)
