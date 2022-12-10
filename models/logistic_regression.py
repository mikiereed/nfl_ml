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

    scores = []
    trials = 5
    for i in range(trials):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        lr = LogisticRegression(C=0.001, penalty='l2', solver='liblinear',)
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        scores.append(accuracy)
        print(f"accuracy: {accuracy}")
        print(f"f1: {f1}")
        # print(lr.coef_)

    print(f"average: {sum(scores) / trials}")


def logitistic_regression_gridcv(data_file: str) -> None:
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=True)
    change_zeros_to_mean(x)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)

    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=2)

    parameters = {
        "solver": ["lbfgs", "newton-cg", "liblinear", ],
        "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100 ],
        "penalty": ["none", "l1", "l2", "elasticnet", ],
    }

    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid=parameters,
        cv=cv,
        scoring=["accuracy", "f1"],
        refit="f1",
        verbose=2,
    )
    grid_search.fit(x_train, y)
    print(f"Score: {grid_search.best_score_}")
    print(f"Parameters: {grid_search.best_params_}")

    # prints
    # Score: 0.85714
    # Parameters: {'C': 0.0001, 'penalty': 'l2', 'solver': 'liblinear'}

if __name__ == "__main__":
    logistic_regression_model("../data/data.csv")
    # logitistic_regression_gridcv("../data/data.csv")
