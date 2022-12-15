import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from model_utils import load_csv, change_zeros_to_mean, normalize_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def logistic_regression_model(data_file: str):
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)

    trials = 100
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(trials):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # lr = LogisticRegression()
        lr = LogisticRegression(C=0.001, penalty='l2', solver='lbfgs', )
        # lr = LogisticRegression(C=0.001, penalty='l2', solver='liblinear', )
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
        precision_scores.append(metrics.precision_score(y_test, y_pred))
        recall_scores.append(metrics.recall_score(y_test, y_pred))

    print(f"accuracy: {sum(accuracy_scores) / trials}")
    print(f"f1: {sum(f1_scores) / trials}")
    print(f"recall: {sum(recall_scores) / trials}")
    print(f"precision: {sum(precision_scores) / trials}")

    # lbfgs
    # accuracy: 0.8216707317073169
    # f1: 0.8474124402081686
    # recall: 0.8856245543692058
    # precision: 0.8127481382227315

    # liblinear
    # accuracy: 0.8244512195121948  <--- winner
    # f1: 0.8415011581858179
    # recall: 0.8336076433099503
    # precision: 0.8499044200450249

    fig = plot_confusion_matrix(lr, x_test, y_test, display_labels=["loss", "win"])
    fig.figure_.suptitle("Confusion Matrix")
    # plt.show()
    plt.savefig("../data/training_confusion_matrix_lr.pdf")


def logitistic_regression_gridcv(data_file: str) -> None:
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=True)
    change_zeros_to_mean(x)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)

    scores = []
    best_params = []
    trials = 20
    for i in range(trials):
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=i)

        parameters = {
            # "solver": ["lbfgs", "newton-cg", "liblinear", ],
            "solver": ["lbfgs", "liblinear", ],
            # "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, ],
            "C": [1e-3, ],
            # "penalty": ["none", "l1", "l2", "elasticnet", ],
            "penalty": ["l2", ],
        }

        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid=parameters,
            cv=cv,
            scoring=["accuracy", "f1"],
            refit="f1",
            verbose=1,
        )
        grid_search.fit(x_train, y)
        # print(f"Score: {grid_search.best_score_}")
        # print(f"Parameters: {grid_search.best_params_}")
        scores.append(grid_search.best_score_)
        best_params.append(grid_search.best_params_)

    c_counts = {1e-5: 0, 1e-4: 0, 1e-3: 0, 1e-2: 0, 1e-1: 0, 1: 0, 10: 0, 100: 0}
    solver_counts = {"lbfgs": 0, "newton-cg": 0, "liblinear": 0}
    penalty_counts = {"none": 0, "l1": 0, "l2": 0, "elasticnet": 0}

    for i in range(trials):
        print(f"{scores[i] = }")
        print(f"{best_params[i] = }")
        c_counts[best_params[i]["C"]] += 1
        solver_counts[best_params[i]["solver"]] += 1
        penalty_counts[best_params[i]["penalty"]] += 1

    print(f"{c_counts = }")
    print(f"{solver_counts = }")
    print(f"{penalty_counts = }")

    # with all parameters allowed prints
    # scores[i] = 0.8513800424628452
    # best_params[i] = {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
    # scores[i] = 0.8468085106382978
    # best_params[i] = {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8571428571428572
    # best_params[i] = {'C': 0.0001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8622881355932204
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8490990990990991
    # best_params[i] = {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8558659217877095
    # best_params[i] = {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
    # scores[i] = 0.837158469945355
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8259911894273128
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8297872340425532
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8523206751054853
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8580310880829015
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8493723849372384
    # best_params[i] = {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8673684210526315
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8553846153846153
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.849846782431052
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8471794871794872
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8494845360824743
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8524590163934427
    # best_params[i] = {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8520084566596194
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8553719008264463
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # c_counts = {1e-05: 0, 0.0001: 1, 0.001: 13, 0.01: 3, 0.1: 3, 1: 0, 10: 0, 100: 0}
    # solver_counts = {'lbfgs': 12, 'newton-cg': 0, 'liblinear': 8}
    # penalty_counts = {'none': 0, 'l1': 2, 'l2': 18, 'elasticnet': 0}

    # with only lbfgs and liblinear params allowed prints
    # scores[i] = 0.8489208633093526
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8459958932238192
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8565217391304347
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8622881355932204
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.843069873997709
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.847385272145144
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.837158469945355
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8259911894273128
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8297872340425532
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8523206751054853
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8580310880829015
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8446700507614213
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8673684210526315
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8553846153846153
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.849846782431052
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8471794871794872
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8494845360824743
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8482834994462902
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
    # scores[i] = 0.8520084566596194
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # scores[i] = 0.8553719008264463
    # best_params[i] = {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'}
    # c_counts = {1e-05: 0, 0.0001: 0, 0.001: 20, 0.01: 0, 0.1: 0, 1: 0, 10: 0, 100: 0}
    # solver_counts = {'lbfgs': 15, 'newton-cg': 0, 'liblinear': 5}
    # penalty_counts = {'none': 0, 'l1': 0, 'l2': 20, 'elasticnet': 0}


if __name__ == "__main__":
    logistic_regression_model("../data/data.csv")
    # logitistic_regression_gridcv("../data/data.csv")
