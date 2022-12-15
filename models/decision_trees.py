from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.model_utils import load_csv, change_zeros_to_mean


def decision_trees_model(data_file: str) -> None:
    """
    :param data_file: file of data set
    :return:
    """
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

        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
        precision_scores.append(metrics.precision_score(y_test, y_pred))
        recall_scores.append(metrics.recall_score(y_test, y_pred))

    print(f"accuracy: {sum(accuracy_scores) / trials}")
    print(f"f1: {sum(f1_scores) / trials}")
    print(f"recall: {sum(recall_scores) / trials}")
    print(f"precision: {sum(precision_scores) / trials}")

    # prints
    # accuracy: 0.7197926829268294
    # f1: 0.7495522403735752
    # recall: 0.7501486714957629
    # precision: 0.7494874879628307


if __name__ == "__main__":
    decision_trees_model("../data/data.csv")
