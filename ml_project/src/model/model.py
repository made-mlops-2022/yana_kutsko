import json
import os
import pickle
import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from transformers.feature_selector import FeatureSelector


class Model:
    def __init__(self, classifier_type: str, categorical_features: List[str],
                 numerical_features: List[str]) -> None:
        self.best_classifier: Union[KNeighborsClassifier, LogisticRegression, None] = None

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        if classifier_type == 'KNeighborsClassifier':
            self.classifier = KNeighborsClassifier()
            self.param_grid = {'model__n_neighbors': range(1, 20),
                               'model__weights': ['uniform', 'distance'],
                               'model__metric': ['euclidean', 'manhattan', 'cosine']}
        elif classifier_type == 'LogisticRegression':
            self.classifier = LogisticRegression(max_iter=10_000, solver='liblinear')
            self.param_grid = {"model__C": np.logspace(-3, 3, 7),
                               "model__penalty": ["l1", "l2"]}
        else:
            raise ValueError('Only KNeighborsClassifier and LogisticRegression are supported')

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:

        categorical_pipeline = Pipeline([
            ('selector', FeatureSelector(self.categorical_features)),
            ('imputer', SimpleImputer())
        ])
        numerical_pipeline = Pipeline([
            ('selector', FeatureSelector(self.numerical_features)),
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ])
        full_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                                       ('numerical_pipeline', numerical_pipeline)])
        full_pipeline_with_model = Pipeline([
            ('transformer', full_pipeline),
            ('model', self.classifier)
        ])
        classifier_search = GridSearchCV(full_pipeline_with_model, self.param_grid, cv=10)
        classifier_search.fit(X_train, y_train)
        self.best_classifier = classifier_search.best_estimator_

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self.best_classifier is None:
            raise ValueError('Train model first')
        return self.best_classifier.predict(X_test)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        predicted_values = self.predict(X_test)
        return {'f1_score': f1_score(y_test, predicted_values),
                'roc_auc_score': roc_auc_score(y_test, predicted_values),
                'accuracy_score': accuracy_score(y_test, predicted_values)}

    def serialize_model(self, path_to_model: str) -> None:
        os.makedirs(os.path.dirname(path_to_model), exist_ok=True)
        with open(path_to_model, 'wb+') as file:
            pickle.dump(self.best_classifier, file)

    def score_report(self, path_to_score: str, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        os.makedirs(os.path.dirname(path_to_score), exist_ok=True)
        with open(path_to_score, 'w+') as file:
            json.dump(self.score(X_test, y_test), file)
