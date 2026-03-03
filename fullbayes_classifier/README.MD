# FullBayes IIT

Full Covariance Gaussian Bayesian Classifier.

## Installation

pip install fullbayes_iit

## Usage

from fullbayes import FullBayesClassifier

clf = FullBayesClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)