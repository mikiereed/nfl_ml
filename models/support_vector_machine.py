from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

from models.model_utils import load_csv, change_zeros_to_mean


def svm_model(data_file: str) -> None:
    """
    :param data_file: file of data set
    :return:
    """
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)

    scores = []
    for i in range(20):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        svm = SVC(C=1, gamma=0.01, kernel='rbf')
        # svm = SVC()
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        print(f"{accuracy}")

    print(f"average: {sum(scores) / 20}")

def svm_gridcv(data_file: str) -> None:
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=True)
    change_zeros_to_mean(x)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)

    for i in range(10):
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=i)

        parameters = {
            'C': [0.1,1, 10, 100, ],
            'gamma': [1,0.1,0.01,0.001, ],
            'kernel': ['rbf', 'poly', 'sigmoid', ],
        }

        grid_search = GridSearchCV(
            SVC(),
            param_grid=parameters,
            cv=cv,
            scoring=["recall", "f1"],
            refit="f1",
            verbose=1,
        )
        grid_search.fit(x_train, y)
        print(f"Score: {grid_search.best_score_}")
        print(f"Parameters: {grid_search.best_params_}")

    # prints
    # Score: 0.8631578947368421
    # Parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

if __name__ == "__main__":
    svm_model("../data/data.csv")
    # svm_gridcv("../data/data.csv")
