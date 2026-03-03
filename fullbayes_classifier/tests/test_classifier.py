import numpy as np
from fullbayes import FullBayesClassifier


def test_fit_predict():
    X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
    y = np.array([0, 0, 1, 1])

    clf = FullBayesClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)

    assert len(preds) == len(y)