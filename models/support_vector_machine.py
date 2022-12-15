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

    trials = 100
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # svm = SVC()
        # svm = SVC(C=1, gamma=0.01, kernel='rbf')
        svm = SVC(C=10, gamma=0.01, kernel='rbf')
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
        precision_scores.append(metrics.precision_score(y_test, y_pred))
        recall_scores.append(metrics.recall_score(y_test, y_pred))

    print(f"accuracy: {sum(accuracy_scores) / trials}")
    print(f"f1: {sum(f1_scores) / trials}")
    print(f"recall: {sum(recall_scores) / trials}")
    print(f"precision: {sum(precision_scores) / trials}")

    # C: 1
    # accuracy: 0.8243292682926832
    # f1: 0.8457712263111737
    # recall: 0.8615558170738129
    # precision: 0.8309451908535403

    # C: 10
    # accuracy: 0.8117682926829266
    # f1: 0.8333839612615157
    # recall: 0.84199053953803
    # precision: 0.825359462629439


def svm_gridcv(data_file: str) -> None:
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
        scores.append(grid_search.best_score_)
        best_params.append(grid_search.best_params_)

    c_counts = {0.1: 0, 1: 0, 10: 0, 100: 0}
    gamma_counts = {1: 0, 0.1: 0, 0.01: 0, 0.001: 0}
    kernel_counts = {'rbf': 0, 'poly': 0, 'sigmoid': 0}

    for i in range(trials):
        print(f"{scores[i] = }")
        print(f"{best_params[i] = }")
        c_counts[best_params[i]["C"]] += 1
        gamma_counts[best_params[i]["gamma"]] += 1
        kernel_counts[best_params[i]["kernel"]] += 1

    print(f"{c_counts = }")
    print(f"{gamma_counts = }")
    print(f"{kernel_counts = }")

if __name__ == "__main__":
    # svm_model("../data/data.csv")
    svm_gridcv("../data/data.csv")
