from model_utils import load_csv, plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def logistic_regression_model(data_file: str):
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=True)
    scores = []
    for i in range(20):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=i)

        logistic_regression = LogisticRegression(max_iter=1_000_000)
        logistic_regression.fit(x_train, y_train)

        theta = logistic_regression.coef_

        # Returns a NumPy Array
        # Predict for One Observation (image)
        predictions = logistic_regression.predict(x_test)
        check = predictions == y_test

        score = logistic_regression.score(x_test, y_test)
        print(f"{i}: {score}")
        scores.append(score)

    print(f"average: {sum(scores) / 20}")

if __name__ == "__main__":
    logistic_regression_model("../data/data.csv")
