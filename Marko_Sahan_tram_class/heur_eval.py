import numpy as np
import model_eval as me
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored


def positive_or_negative():
    if np.random.uniform() < 0.5:
        return 1
    else:
        return -1


def crossover(first_parent, second_parent):

    offspring: list = []
    for feature in range(5):
        if positive_or_negative == 1:
            offspring.append(first_parent[feature])
        else:
            offspring.append(second_parent[feature])

    return offspring


def generate_random_parameters(parameters):

    bootstrap = parameters.bootstrap[int(np.round(np.random.uniform()))]
    criterion = parameters.criterion[int(np.round(np.random.uniform()))]
    n_estimators = parameters.n_estimators[int(
        np.round(np.random.uniform()*199))]
    max_depth = parameters.max_depth[int(np.round(np.random.uniform()*29))]
    max_features = parameters.max_features[int(
        np.round(np.random.uniform()*3))]

    return (bootstrap,
            criterion,
            n_estimators,
            max_depth,
            max_features)


def mutate(parameters,
           bootstrap,
           criterion,
           n_estimators,
           max_depth,
           max_features):

    mutated_feature = int(np.round(np.random.uniform()*4))
    if mutated_feature == 0:
        bootstrap = parameters.bootstrap[int(np.round(np.random.uniform()))]
    if mutated_feature == 1:
        criterion = parameters.criterion[int(np.round(np.random.uniform()))]
    if mutated_feature == 2:
        n_estimators = parameters.n_estimators[int(
            np.round(np.random.uniform()*199))]
    if mutated_feature == 3:
        max_depth = parameters.max_depth[int(np.round(np.random.uniform()*29))]
    if mutated_feature == 4:
        max_features = parameters.max_features[int(
            np.round(np.random.uniform()*3))]

    return (bootstrap,
            criterion,
            n_estimators,
            max_depth,
            max_features)


def generate_neighbor(parameters: me.randomForestParameters,
                      bootstrap,
                      criterion,
                      n_estimators,
                      max_depth,
                      max_features,
                      windowsize):

    bootstrap = parameters.bootstrap[int(np.round(np.random.uniform()))]
    criterion = parameters.criterion[int(np.round(np.random.uniform()))]
    max_features = parameters.max_features[int(
        np.round(np.random.uniform()*3))]

    step = int(np.round(np.random.uniform()*windowsize))
    n_est_index = parameters.n_estimators.index(n_estimators)
    if n_est_index == 0 or n_est_index - windowsize < 0:
        n_estimators = parameters.n_estimators[n_est_index+step]
    elif n_est_index == 199 or n_est_index + windowsize > 29:
        n_estimators = parameters.n_estimators[n_est_index-step]
    else:
        n_estimators = parameters.n_estimators[n_est_index +
                                               positive_or_negative()*step]

    step = int(np.round(np.random.uniform()*windowsize))
    max_depth_index = parameters.max_depth.index(max_depth)
    if max_depth_index == 0 or max_depth_index - windowsize < 0:
        max_depth = parameters.max_depth[max_depth_index+step]
    elif max_depth_index == 29 or max_depth_index + windowsize > 29:
        max_depth = parameters.max_depth[max_depth_index-step]
    else:
        max_depth = parameters.max_depth[max_depth_index +
                                         positive_or_negative()*step]

    return (bootstrap,
            criterion,
            n_estimators,
            n_estimators,
            max_depth,
            max_features)
