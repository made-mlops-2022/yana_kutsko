import os
import pickle

import click
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option("--features_train_path", required=True)
@click.option("--target_train_path", required=True)
@click.option("--model_path", required=True)
def main(features_train_path: str,
         target_train_path: str,
         model_path) -> None:
    X_train: np.ndarray
    y_train: np.ndarray

    X_train = np.genfromtxt(features_train_path, delimiter=',')
    y_train = np.genfromtxt(target_train_path, delimiter=',')

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb+') as file:
        pickle.dump(clf, file)


if __name__ == '__main__':
    main()
