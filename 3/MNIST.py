import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
assert sklearn.__version__ >= "0.20"
assert sys.version_info >= (3, 5)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
np.random.seed(42)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', data_home='./datasets', version=1, cache=True)
    # print(mnist.keys())
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    print(y[0])
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    sgd_clf.predict(X[0])
    # some_digit = X[1]
    # some_digit_image = some_digit.reshape(28, 28)
    # plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    # plt.axis("off")
    #
    # save_fig("some_digit_plot")
    # plt.show()