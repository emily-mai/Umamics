import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


def gmm_anomalies(data, posterior, threshold):
    max_posterior = np.amax(posterior, axis=1)
    indices = np.argwhere(max_posterior < threshold).flatten()
    anomalies = data["text"].take(indices)
    return anomalies


def lof_anomalies(data, X):
    model = LocalOutlierFactor(contamination=0.01, n_neighbors=5, metric="l2")
    outlier_scores = model.fit_predict(X)
    print(model.negative_outlier_factor_[outlier_scores == -1])
    anomalies = data[outlier_scores == -1]
    return anomalies["text"]


def dbscan_anomalies(data, labels):
    indices = np.argwhere(labels == -1).flatten()
    anomalies = data["text"].take(indices)
    return anomalies


if __name__ == "__main__":
    # load data for one business
    data = pd.read_pickle("data/data.pkl")
    data = data[data["business_id"] == "bZiIIUcpgxh8mpKMDhdqbA"]
    X = np.array(data["embedding"].tolist())

    # fit and predict models on data
    best_dbscan = DBSCAN(eps=0.85, min_samples=7).fit(X)
    best_gmm = GaussianMixture(n_components=4, covariance_type='diag').fit(X)
    Y = best_gmm.predict(X)

    anomalies_gmm = gmm_anomalies(data, best_gmm.predict_proba(X), 0.99)
    anomalies_dbscan = dbscan_anomalies(data, best_dbscan.labels_)
    anomalies_lof = lof_anomalies(data, X)
