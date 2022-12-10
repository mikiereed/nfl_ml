from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from model_utils import load_csv, change_zeros_to_mean, normalize_features


def mlp_classification(data_file: str) -> None:
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)
    # normalize_features(x)

    scores = []
    params = []
    for i in range(20):
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)
        x_train = scaler.fit_transform(x)

        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=i * 4)

        parameters = {
            # "hidden_layer_sizes": [5, 10, 15, (5, 5), (5, 10)],
            "hidden_layer_sizes": [10, ],
            # "activation": ["identity", "logistic", "tanh", "relu", ],
            "activation": ["logistic",],
            "solver": ["adam", ],
            # "learning_rate": ["constant", "adaptive", ],
            "learning_rate": ["constant", ],
            # "alpha": [0, 0.0005, 0.0001, 0.005, 0.001, ],
            "alpha": [0.001, ],
            # "max_iter": [200, 350, 500, 1000, ],
            "max_iter": [350, ],
            # "shuffle": [True, False, ],
            "shuffle": [True, ],
        }

        grid_search = GridSearchCV(
            MLPClassifier(),
            param_grid=parameters,
            cv=cv,
            scoring=["recall", "f1"],
            refit="f1",
            verbose=0,
        )
        grid_search.fit(x_train, y)
        print(f"Score: {grid_search.best_score_}")
        print(f"Parameters: {grid_search.best_params_}")
        scores.append(grid_search.best_score_)
        params.append(grid_search.best_params_)

    for i in range(len(scores)):
        print(f"Score: {scores[i]}")
        print(f"Parameters: {params[i]}")

    # clf = MLPClassifier(
    #     hidden_layer_sizes=(256, 128, 64, 32),
    #     activation="relu",
    #     random_state=1)\
    #     .fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print(clf.score(x_test, y_test))
    #
    # fig = plot_confusion_matrix(clf, x_test, y_test, display_labels=["loss", "win"])
    # fig.figure_.suptitle("Confusion Matrix")
    # plt.show()


if __name__ == "__main__":
    mlp_classification("../data/data.csv")
