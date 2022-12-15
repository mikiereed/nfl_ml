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


def final_model(training_data_file: str, final_test_data_file: str) -> None:
    x_train, y_train = load_csv(csv_path=training_data_file, label_col="y", add_intercept=False)
    x_test, y_test = load_csv(csv_path=final_test_data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x_train)
    change_zeros_to_mean(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    lr = LogisticRegression(C=0.001, penalty='l2', solver='liblinear',)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print(f"accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"f1: {metrics.f1_score(y_test, y_pred)}")
    print(f"precision: {metrics.precision_score(y_test, y_pred)}")
    print(f"recall: {metrics.recall_score(y_test, y_pred)}")

    # prints
    # accuracy: 0.7794871794871795
    # f1: 0.7942583732057417
    # precision: 0.8137254901960784
    # recall: 0.7757009345794392

    theta = lr.coef_
    print(f"{len(theta[0]) = }")
    print(f"{theta = }")

    # prints
    # len(theta[0]) = 36
    # theta = array([[0.03000381, 0.03135045, 0.01710864, 0.07395856, 0.02883529,
    #                 0.12723225, 0.09787044, 0.27825057, 0.24138426, 0.07274476,
    #                 0.03428659, 0.13589698, 0.21927479, 0.16047564, 0.04025631,
    #                 -0.00929909, 0.01406469, 0.0280293, -0.05205342, -0.05863057,
    #                 -0.03668082, -0.07252047, -0.0160126, -0.13065325, -0.10343585,
    #                 -0.26035686, -0.20917193, -0.08104996, -0.02460227, -0.12779833,
    #                 -0.20738035, -0.15199249, -0.05894768, -0.00367673, 0.01415094,
    #                 -0.03951392]])

    fig = plot_confusion_matrix(lr, x_test, y_test, display_labels=["loss", "win"])
    fig.figure_.suptitle("Confusion Matrix for 2022 NFL Season Weeks 1-13")
    # plt.show()
    plt.savefig("../data/FINAL_TEST_confusion_matrix.jpg")


if __name__ == "__main__":
    final_model(training_data_file="../data/data.csv", final_test_data_file="../data/FINAL_TEST_data.csv")
