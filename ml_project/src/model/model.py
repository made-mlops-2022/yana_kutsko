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
from ml_project.src.transformers.feature_selector import FeatureSelector
from ml_project.src.loggers.logger import create_root_loger


class Model:
    def __init__(self, classifier_type: str, categorical_features: List[str],
                 numerical_features: List[str]) -> None:
        self.logger = create_root_loger()
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
            self.logger.critical(f'Unsupported ({classifier_type}) model type was selected')
            raise ValueError('Only KNeighborsClassifier and LogisticRegression are supported')

        self.logger.info(f'{classifier_type} model type was selected')

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
        self.logger.debug('Pipeline setup is finished')
        self.logger.debug(f'GridSearchCV with next params starts fitting: {self.param_grid}')
        classifier_search = GridSearchCV(full_pipeline_with_model, self.param_grid, cv=10)
        classifier_search.fit(X_train, y_train)
        self.logger.info(f'GridSearchCV with next params successfully finished fitting process: {self.param_grid}')
        self.best_classifier = classifier_search.best_estimator_
        self.logger.info(f'Best estimator score is {classifier_search.best_score_}')

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self.best_classifier is None:
            self.logger.critical('Model is not trained')
            raise ValueError('Train model first')
        self.logger.debug('Model starts making predictions')
        return self.best_classifier.predict(X_test)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        predicted_values = self.predict(X_test)
        score = {'f1_score': f1_score(y_test, predicted_values),
                 'roc_auc_score': roc_auc_score(y_test, predicted_values),
                 'accuracy_score': accuracy_score(y_test, predicted_values)}
        return score

    def serialize_model(self, path_to_model: str) -> None:
        os.makedirs(os.path.dirname(path_to_model), exist_ok=True)
        with open(path_to_model, 'wb+') as file:
            pickle.dump(self.best_classifier, file)
            self.logger.info(f'Model was serialized successfully and stored in {path_to_model}')

    def score_report(self, path_to_score: str, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        os.makedirs(os.path.dirname(path_to_score), exist_ok=True)
        with open(path_to_score, 'w+') as file:
            json.dump(self.score(X_test, y_test), file)
            self.logger.info(f'Score report was created successfully and stored in {path_to_score}')
