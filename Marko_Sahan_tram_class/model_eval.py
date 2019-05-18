import sklearn
import numpy as np
import pickle

from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class randomForestParameters:
    def __init__(self):
        self.bootstrap = [True, False]
        self.criterion = ['gini', 'entropy']
        self.n_estimators = [value for value in range(10, 211)]
        self.max_depth = [value for value in range(2, 33)]
        self.max_features = ['auto', 'sqrt', 'log2', None]


def loss_function(test_vals: np.ndarray, pred_vals: np.ndarray) -> float:

    zero_one_loss = 1 / \
        len(test_vals)*np.sum([1 for val in (test_vals-pred_vals) if val != 0])

    return zero_one_loss


def save_model(model, file_name: str):
    pickle.dump(model, open(file_name, 'wb'))


def calculate_statistics(data):

    alg_numpy = np.asmatrix(data)
    alg_mean = np.mean(alg_numpy, axis=0)
    alg_upper = np.max(data, axis=0)
    alg_lower = np.min(data, axis=0)

    alg_mean = alg_mean.tolist()[0]
    alg_upper = alg_upper.tolist()[0]
    alg_lower = alg_lower.tolist()[0]

    return alg_mean, alg_upper, alg_lower
