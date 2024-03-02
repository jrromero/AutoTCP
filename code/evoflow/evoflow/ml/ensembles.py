# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import numpy as np

from sklearn.preprocessing import LabelEncoder


class VotingClassifier:

    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.label_encoder = None

    def fit(self, X, y):
        self.classes_ = True
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

    def predict(self, X):

        # Differentiate between different voting schemas
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)
        elif self.voting == 'hard':
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.estimators]).T
            predictions = np.apply_along_axis(
                self.label_encoder.transform, 1, predictions)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions)

        # Convert integer predictions to original labels:
        return self.label_encoder.inverse_transform(maj)

    def predict_proba(self, X):

        # Differentiate between different voting schemas
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        predictions = np.asarray([clf.predict_proba(X)
                                  for clf in self.estimators])
        avg = np.average(predictions, axis=0, weights=self.weights)
        return avg