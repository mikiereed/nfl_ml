from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from model_utils import load_csv, change_zeros_to_mean, normalize_features


def mlp_classification(data_file: str) -> None:
    """
    Idea for most of this code from https://www.youtube.com/watch?v=oPxbrgYuBGY
    :param data_file: filename of csv data set
    :return:
    """
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=True)
    change_zeros_to_mean(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
    #
    # parameters = {
    #     "hidden_layer_sizes": [5, 10, 15, (5, 5), (5, 10)],
    #     "activation": ["identity", "logistic", "tanh", "relu", ],
    #     "solver": ["lbfgs", "sgd", "adam", ],
    #     "learning_rate": ["constant", "invscaling", "adaptive", ],
    #     "alpha": [0, 0.0005, 0.0001, 0.005, 0.001, ],
    #     "max_iter": [200, 500],
    #     "shuffle": [True, False, ],
    # }
    #
    # grid_search = GridSearchCV(
    #     MLPClassifier(),
    #     param_grid=parameters,
    #     cv=cv,
    #     scoring=["recall", "f1"],
    #     refit="f1",
    #     verbose=2,
    # )
    # grid_search.fit(x_train, y_train)
    # print(f"Score: {grid_search.best_score_}")
    # print(f"Parameters: {grid_search.best_params_}")

    mlp = MLPClassifier(
        hidden_layer_sizes=15,
        activation="logistic",
        alpha=0.005,
        learning_rate="adaptive",
        max_iter=500,
        shuffle=True,
        solver="adam",
        random_state=1)\
        .fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    print(mlp.score(x_test, y_test))  # y_pred???

    fig = plot_confusion_matrix(mlp, x_test, y_test, display_labels=["loss", "win"])
    fig.figure_.suptitle("Confusion Matrix")
    # plt.show()
    plt.savefig("../data/confusion_matrix.pdf")


if __name__ == "__main__":
    mlp_classification("../data/data.csv")
