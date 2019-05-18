import numpy as np
import model_eval as me
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
import heur_eval as he

''' 
Here is represented approach for Steepest Descent heuristics. 
Due to computational costs it was simplified to the form that is shown here. 
This heuristics randomly chooses parameters for Random Forest and then randomly
generated parameters that are in the neighborhood of the initial parameters. If that 
parameter has better metrics than heuristics switches generates new set of parameters 
in a neighborhood of the new set of parameters... Algorithm iterates through 100 iterations. 
'''


def steepest_descent(data: np.ndarray,
                     labels: np.ndarray,
                     parameters: me.randomForestParameters,
                     loss_function: Callable,
                     window_size: int):

    # define amount of iterations and initial loss
    iterations = 100
    loss_function_evolution: list = []
    split = int(np.round(0.75*len(data)))

    # create training and testimg dataset by splitting as 75% of the data are taining data 25% are tesing data
    training_data = data[:split, :]
    training_labels = labels[:split]

    testing_data = data[split:, :]
    testing_labels = labels[split:]

    # generate initial set of parameters and calculate metrics
    (bootstrap,
     criterion,
     n_estimators,
     max_depth,
     max_features) = he.generate_random_parameters(parameters)

    # save that parameters because in next steps we are going to generate new parameters and compare metrics
    bootstrap_old, criterion_old, n_estimators_old, max_depth_old, max_features_old = bootstrap, criterion, n_estimators, max_depth, max_features

    clf = RandomForestClassifier(bootstrap=bootstrap_old,
                                 criterion=criterion_old,
                                 n_estimators=n_estimators_old,
                                 max_depth=max_depth_old,
                                 max_features=max_features_old,
                                 random_state=0)

    clf.fit(training_data, training_labels)
    predicted_labels = clf.predict(testing_data)

    loss = loss_function(testing_labels, predicted_labels)
    best_accuracy = accuracy_score(testing_labels, predicted_labels)
    best_params = [bootstrap,
                   criterion,
                   n_estimators,
                   max_depth,
                   max_features]

    # iterate through iterations
    for _ in range(iterations):

        # generate new parameters and calculate metrics
        (bootstrap,
         criterion,
         n_estimators,
         n_estimators,
         max_depth,
         max_features) = he.generate_neighbor(parameters,
                                              bootstrap,
                                              criterion,
                                              n_estimators,
                                              max_depth,
                                              max_features,
                                              window_size)

        clf = RandomForestClassifier(bootstrap=bootstrap,
                                     criterion=criterion,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     max_features=max_features,
                                     random_state=0)

        clf.fit(training_data, training_labels)
        predicted_labels = clf.predict(testing_data)

        new_loss = loss_function(testing_labels, predicted_labels)

        # if new loss is lover than previous -> continue with new parameters. If not -> continue with old parameters
        if new_loss < loss:
            loss = new_loss
            best_accuracy = accuracy_score(testing_labels, predicted_labels)
            best_params = [bootstrap,
                           criterion,
                           n_estimators,
                           max_depth,
                           max_features]

            bootstrap_old = bootstrap
            criterion_old = criterion
            n_estimators_old = n_estimators
            max_depth_old = max_depth
            max_features_old = max_features

            print(colored(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(accuracy_score(testing_labels, predicted_labels),3)}", "green"))
        else:
            bootstrap = bootstrap_old
            criterion = criterion_old
            n_estimators = n_estimators_old
            max_depth = max_depth_old
            max_features = max_features_old

            print(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(accuracy_score(testing_labels, predicted_labels),3)}")

        loss_function_evolution.append(np.round(loss, 3))

    print(" ")
    print(colored(
        f"The best loss is {np.round(loss,3)}. The best accuracy is {np.round(best_accuracy,3)}", "blue"))
    print(" ")
    return best_params, loss_function_evolution
