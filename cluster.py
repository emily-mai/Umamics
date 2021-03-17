import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


# clustering using Gaussian Mixture Models
# grid search hyper-parameters and return best model and scores
def gmm_model_selection(n_clusters, X):
    lowest_bic = np.infty
    bic = []
    assert X.shape[0] >= n_clusters
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for k in range(1, n_clusters):
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=k, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print("Num components chosen by GMM: {}".format(best_gmm.n_components))
    return best_gmm, bic


def anamoly_detection(posterior, threshold):
    n, k = posterior.shape
    max_posterior = np.amax(posterior, axis=1)
    # TODO: get anomalies


def plot_tsne(gmm, X, Y):
    # dimensionality reduction
    x_reduced = TSNE(n_components=2).fit_transform(X)

    # plot points and their respective colors according to cluster
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    for i, (mean, color) in enumerate(zip(gmm.means_, color_iter)):
        if not np.any(Y == i):
            continue
        plt.scatter(x_reduced[Y == i, 0], x_reduced[Y == i, 1], .8, color=color)

    plt.xticks(())
    plt.yticks(())
    plt.title(f'Selected GMM: {gmm.covariance_type} model, 'f'{gmm.n_components} components')
    plt.show()


# clustering using DBSCAN
# grid search hyper-parameters and return best model and scores
def dbscan_model_selection(X):
    lowest_ss = np.infty
    scores = []
    for epsilon in [.95, 1]:
        for min_samples in [3, 5, 7, 9]:
            db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
            labels = db.labels_
            scores.append(metrics.silhouette_score(X, labels))
            if scores[-1] < lowest_ss:
                lowest_ss = scores[-1]
                best_db = db
    labels = best_db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(lowest_ss, n_clusters_, n_noise_)
    return best_db, scores


if __name__ == "__main__":
    # load data for one business
    data = pd.read_pickle("data/data.pkl")
    data = data[data["business_id"] == "v1UzkU8lEWdjxq8byWFOKg"]
    X = np.array(data["embedding"].tolist())

    # run once to get best model
    # best_gmm, bic = gmm_model_selection(10, X)

    # fit and predict GMM on data
    best_gmm = GaussianMixture(n_components=4, covariance_type='diag').fit(X)
    Y = best_gmm.predict(X)

    # get posterior distribution of p(yi=c|xi) for each cluster c and data xi
    posterior = best_gmm.predict_proba(X)
    anamoly_detection(posterior, threshold=0.1)

    # generate plot using TSNE
    # plot_tsne(best_gmm, X, Y)

    # best_db, scores = dbscan_model_selection(X)
