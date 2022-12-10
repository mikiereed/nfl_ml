from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from models.model_utils import load_csv, change_zeros_to_mean


def visualize_pca(data_file: str) -> None:
    """
    Idea for code comes from https://medium.com/towards-data-science/pca-using-python-scikit-learn-e653f8989e60
    :param data_file: file of csv data
    :return:
    """
    x, y = load_csv(csv_path=data_file, label_col="y", add_intercept=False)
    change_zeros_to_mean(x)

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(
        data = principal_components,
        columns = ['principal component 1', 'principal component 2'],
    )

    pca_df = pd.concat([principal_df, pd.DataFrame(data=y, columns=["outcome"])], axis=1)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=20)
    ax.set_ylabel('Principal Component 2', fontsize=20)
    ax.set_title('2 component PCA', fontsize=25)
    outcomes = [0, 1,]
    colors = ['r', 'b',]
    for outcome, color in zip(outcomes, colors):
        indices = pca_df['outcome'] == outcome
        ax.scatter(
            pca_df.loc[indices, 'principal component 1'],
            pca_df.loc[indices, 'principal component 2'],
            c=color,
            s=50,
        )
    ax.legend(["Loss", "Win", ], fontsize=20)
    ax.grid()
    # plt.show()
    plt.savefig("../data/pca_visual.pdf")

    print(pca.explained_variance_ratio_)


if __name__ == "__main__":
    visualize_pca("../data/data.csv")