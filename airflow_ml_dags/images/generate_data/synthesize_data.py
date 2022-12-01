import os
import random

import click
import numpy as np
from sklearn.datasets import make_classification


@click.command()
@click.option("--features_path", required=True)
@click.option("--target_path", required=True)
def main(features_path: str, target_path: str) -> None:
    n_samples: int = random.randint(100, 1000)

    features: np.ndarray
    target: np.ndarray
    features, target = make_classification(n_samples=n_samples, n_features=10, n_informative=5)

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    np.savetxt(features_path, features, delimiter=",")
    np.savetxt(target_path, target, delimiter=",")


if __name__ == '__main__':
    main()
