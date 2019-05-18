import numpy as np
import model_eval as me
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from termcolor import colored
import heur_eval as he


'''
Here is represented Genetic Optimization heuristics that is based on random crossover function. 
Mean value of the genes that offspring get from parents is three genes from first and three 
genes from second parent. 
'''


def geneteic_optimization(data: np.ndarray,
                          labels: np.ndarray,
                          parameters: me.randomForestParameters,
                          loss_function: Callable,
                          number_of_parents: int = 5,
                          mutation_par: float = 0.3):

    # define amount of iterations and initial loss
    iterations = 100
    loss = 100

    # define empty list for parents, loss and accuracy
    parents: list = []
    parents_loss: list = []
    parents_accuracy: list = []

    loss_function_evolution: list = []

    # create training and testimg dataset by splitting as 75% of the data are taining data 25% are tesing data
    split = int(np.round(0.75*len(data)))

    training_data = data[:split, :]
    training_labels = labels[:split]

    testing_data = data[split:, :]
    testing_labels = labels[split:]

    # create first population and calculate metrics for each parent
    for parent in range(number_of_parents):

        parents.append([he.generate_random_parameters(parameters)])

        clf = RandomForestClassifier(bootstrap=parents[parent][0][0],
                                     criterion=parents[parent][0][1],
                                     n_estimators=parents[parent][0][2],
                                     max_depth=parents[parent][0][3],
                                     max_features=parents[parent][0][4],
                                     random_state=0)
        clf.fit(training_data, training_labels)
        predicted_labels = clf.predict(testing_data)
        parents_loss.append(loss_function(testing_labels, predicted_labels))
        parents_accuracy.append(accuracy_score(
            testing_labels, predicted_labels))

    # iterate through iterations
    for _ in range(iterations):

        # find two fittest parents get metrics
        fittest = [parents_loss.index(key) for key in sorted(parents_loss)[:2]]
        new_loss = parents_loss[fittest[0]]
        best_accuracy_from_pop = parents_accuracy[fittest[0]]

        if new_loss < loss:
            loss = new_loss
            best_accuracy = best_accuracy_from_pop
            best_params = (parents[fittest[0]][0][0],
                           parents[fittest[0]][0][1],
                           parents[fittest[0]][0][2],
                           parents[fittest[0]][0][3],
                           parents[fittest[0]][0][4]) 

            print(colored(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(best_accuracy_from_pop,3)}", "green"))
        else:
            print(
                f"The best loss is {np.round(loss,3)}. Accuracy for current model is {np.round(best_accuracy_from_pop,3)}")

        loss_function_evolution.append(loss)

        # prepare two parents for a crossover
        first_parent = parents[fittest[0]][0]
        second_parent = parents[fittest[1]][0]

        parents: list = []
        parents_loss: list = []
        parents_accuracy: list = []

        # create the offspring with crossover function
        for parent in range(number_of_parents):

            (bootstrap,
             criterion,
             n_estimators,
             max_depth,
             max_features) = he.crossover(first_parent, second_parent)

            # initialize random mutation
            if np.random.uniform() < mutation_par:
                (bootstrap,
                 criterion,
                 n_estimators,
                 max_depth,
                 max_features) = he.mutate(parameters,
                                           bootstrap,
                                           criterion,
                                           n_estimators,
                                           max_depth,
                                           max_features)

            # get all metrics
            clf = RandomForestClassifier(bootstrap=bootstrap,
                                         criterion=criterion,
                                         n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         max_features=max_features,
                                         random_state=0)

            clf.fit(training_data, training_labels)
            predicted_labels = clf.predict(testing_data)
            parents.append([(bootstrap,
                             criterion,
                             n_estimators,
                             max_depth,
                             max_features)])
            parents_loss.append(loss_function(
                testing_labels, predicted_labels))
            parents_accuracy.append(accuracy_score(
                testing_labels, predicted_labels))

    print(" ")
    print(colored(
        f"The best loss is {np.round(loss,3)}. The best accuracy is {np.round(best_accuracy,3)}", "blue"))
    print(" ")
    return best_params, loss_function_evolution
