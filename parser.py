import numpy as np
from sklearn.datasets import load_svmlight_file

class Parser:
    def __init__(self, target):
        self._data = load_svmlight_file(target, n_features=123)

    def transform_x(self):
        X, _ = self._data
        X = X.toarray()
        v = np.ones(X.shape[0])
        p = np.vstack((v, X.T)).T
        return p

    def transform_y(self):
        # contents = list(map(lambda x: x[0], self._data))
        _, Y = self._data
        return (Y + 1) // 2 # transform into {0, 1}