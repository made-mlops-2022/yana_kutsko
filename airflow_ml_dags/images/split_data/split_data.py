import os
import click
import numpy as np
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--features_preprocessed_path", required=True)
@click.option("--target_path", required=True)
@click.option("--features_train_path", required=True)
@click.option("--features_test_path", required=True)
@click.option("--target_train_path", required=True)
@click.option("--target_test_path", required=True)
def main(features_preprocessed_path: str,
         target_path: str,
         features_train_path: str,
         features_test_path: str,
         target_train_path: str,
         target_test_path: str) -> None:
    features: np.ndarray = np.genfromtxt(features_preprocessed_path, delimiter=',')
    target: np.ndarray = np.genfromtxt(target_path, delimiter=',')

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train, X_test, y_train, y_test = train_test_split(features, target)

    os.makedirs(os.path.dirname(features_train_path), exist_ok=True)

    np.savetxt(features_train_path, X_train, delimiter=",")
    np.savetxt(features_test_path, X_test, delimiter=",")
    np.savetxt(target_train_path, y_train, delimiter=",")
    np.savetxt(target_test_path, y_test, delimiter=",")


if __name__ == '__main__':
    main()
