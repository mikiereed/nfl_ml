from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
        # scaler = StandardScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)

        svm = SVC()
        svm.fit(x_train, y_train)
        y_pred = svm.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        print(f"{accuracy}")


if __name__ == "__main__":
    svm_model("../data/data.csv")