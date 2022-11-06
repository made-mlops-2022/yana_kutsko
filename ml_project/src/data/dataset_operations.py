import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class DatasetOperations:
    @staticmethod
    def read_dataset(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def split_to_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df.iloc[:, :-1], df.iloc[:, -1]

    @staticmethod
    def split_to_train_and_test(features: pd.DataFrame, target: pd.Series, test_size: float,
                                random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(features, target, test_size=test_size, random_state=random_state)
