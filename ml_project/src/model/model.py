import numpy as np
import typing
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class Model:
    def __init__(self, classifier_type: str) -> None:
        self.best_classifier: typing.Union[KNeighborsClassifier, LogisticRegression, None] = None
        if classifier_type == 'KNeighborsClassifier':
            self.classifier = KNeighborsClassifier()
            self.param_grid = {'n_neighbours': range(1, 20),
                               'weights': ['uniform', 'distance'],
                               'metric': ['euclidean', 'manhattan']}
        if classifier_type == 'LogisticRegression':
            self.classifier = LogisticRegression()
            self.param_grid = {"C": np.logspace(-3, 3, 7),
                               "penalty": ["l1", "l2"]}
        else:
            raise ValueError('Only KNeighborsClassifier and LogisticRegression are supported')

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        classifier_cv = GridSearchCV(self.classifier, self.param_grid, scoring='f1', cv=10)
        classifier_cv.fit(X_train, y_train)
        self.best_classifier = classifier_cv.best_estimator_

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self.best_classifier is None:
            raise ValueError('Train model first')
        return self.best_classifier.predict(X_test)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        predicted_values = self.predict(X_test)
        return {'f1_score': f1_score(y_test, predicted_values),
                'roc_auc_score': roc_auc_score(y_test, predicted_values),
                'accuracy_score': accuracy_score(y_test, predicted_values)}

    def serialize_model(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.best_classifier, f)

