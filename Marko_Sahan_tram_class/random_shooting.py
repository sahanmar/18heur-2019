import numpy as np
import model_eval as me
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
import heur_eval as he

''' 
Here is represented Random Shooting heuristics.
'''


def random_shooting(data: np.ndarray,
                    labels: np.ndarray,
                    parameters: me.randomForestParameters,
                    loss_function: Callable):

    # define amount of iterations and initial loss
    iterations = 100
    loss = 100
    loss_function_evolution: list = []

    # create training and testimg dataset by splitting as 75% of the data are taining data 25% are tesing data
    split = int(np.round(0.75*len(data)))

    training_data = data[:split, :]
    training_labels = labels[:split]

    testing_data = data[split:, :]
    testing_labels = labels[split:]

    # iterate through iterations
    for _ in range(iterations):

        # generate random parameters and calculate their metrics
        (bootstrap,
         criterion,
         n_estimators,
         max_depth,
         max_features) = he.generate_random_parameters(parameters)

        clf = RandomForestClassifier(bootstrap=bootstrap,
                                     criterion=criterion,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     max_features=max_features,
                                     random_state=0)

        clf.fit(training_data, training_labels)
        predicted_labels = clf.predict(testing_data)

        new_loss = loss_function(testing_labels, predicted_labels)
        if new_loss < loss:
            loss = new_loss
            best_accuracy = accuracy_score(testing_labels, predicted_labels)
            best_params = [bootstrap,
                           criterion,
                           n_estimators,
                           max_depth,
                           max_features]
            print(colored(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(accuracy_score(testing_labels, predicted_labels),3)}", "green"))
        else:
            print(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(accuracy_score(testing_labels, predicted_labels),3)}")

        loss_function_evolution.append(np.round(loss, 3))
    print(" ")
    print(colored(
        f"The best loss is {np.round(loss,3)}. The best accuracy is {np.round(best_accuracy,3)}", "blue"))
    print(" ")
    return best_params, loss_function_evolution
