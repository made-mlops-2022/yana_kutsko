import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from logger import create_root_loger


class DatasetOperations:
    def __init__(self):
        self.logger = create_root_loger()

    def read_dataset(self, path: str) -> pd.DataFrame:
        self.logger.info(f'Dataset on path {path} starts reading')
        df = pd.read_csv(path)
        self.logger.info(f'Dataset on path {path} is read')
        return df

    def split_to_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info(f'Dataframe starts splitting into features and target')
        features, target = df.iloc[:, :-1], df.iloc[:, -1]
        self.logger.info(f'Dataframe was successfully split')
        return features, target

    def split_to_train_and_test(self, features: pd.DataFrame, target: pd.Series, test_size: float,
                                random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        self.logger.info(f'Data starts splitting into train and test. Test size is {test_size}')
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
        self.logger.info(f'Data was successfully split')
        return X_train, X_test, y_train, y_test
