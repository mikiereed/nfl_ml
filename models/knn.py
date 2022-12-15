import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from models.model_utils import load_csv, change_zeros_to_mean


def knn_model_find_best_k(data_file: str) -> None:
    """
    Code idea from https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75
    :param data_file: file of data set
    :return:
    """
    k_count = 60
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    scores = []
    for k in range(1, k_count):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(range(1, k_count), scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Testing Accuracy")
    # plt.show()
    plt.savefig("../data/knn_training.pdf")


def knn_model(data_file) -> None:
    """
    test knn_model multiple times to get an average accuracy
    :param data_file: file of data set
    :return:
    """
    k = 21
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)

    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    trials = 100
    for i in range(trials):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
        precision_scores.append(metrics.precision_score(y_test, y_pred))
        recall_scores.append(metrics.recall_score(y_test, y_pred))

    print(f"accuracy: {sum(accuracy_scores) / trials}")
    print(f"f1: {sum(f1_scores) / trials}")
    print(f"recall: {sum(recall_scores) / trials}")
    print(f"precision: {sum(precision_scores) / trials}")

    # prints
    # accuracy: 0.8006341463414632
    # f1: 0.8315326232206606
    # recall: 0.8801078187448284
    # precision: 0.7884736098290235


if __name__ == "__main__":
    knn_model_find_best_k("../data/data.csv")
    knn_model("../data/data.csv")
