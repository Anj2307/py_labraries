import numpy as np
from scipy.stats import multivariate_normal


class FullBayesClassifier:
    """
    Full Covariance Gaussian Bayesian Classifier
    """

    def __init__(self):
        self.classes = None
        self.mean = {}
        self.cov = {}
        self.prior = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate class priors, means, and covariance matrices.
        """
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.prior[c] = X_c.shape[0] / X.shape[0]
            self.mean[c] = np.mean(X_c, axis=0)
            self.cov[c] = np.cov(X_c.T)

        return self

    def predict_proba(self, X: np.ndarray):
        """
        Compute posterior probabilities.
        """
        numerators = []

        for c in self.classes:
            likelihood = multivariate_normal.pdf(
                X,
                mean=self.mean[c],
                cov=self.cov[c],
                allow_singular=True
            )
            numerators.append(likelihood * self.prior[c])

        numerators = np.array(numerators)
        total_prob = np.sum(numerators, axis=0)
        posteriors = numerators / total_prob

        return posteriors.T  # shape (n_samples, n_classes)

    def predict(self, X: np.ndarray):
        """
        Predict class labels.
        """
        posteriors = self.predict_proba(X)
        return self.classes[np.argmax(posteriors, axis=1)]