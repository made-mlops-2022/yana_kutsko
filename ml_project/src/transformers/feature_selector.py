from __future__ import annotations
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def fit(self, X, y=None) -> FeatureSelector:
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return X.loc[:, self.feature_names]
