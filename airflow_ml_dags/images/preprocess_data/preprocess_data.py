import os
import click
import numpy as np


@click.command()
@click.option("--features_raw_path", required=True)
@click.option("--features_preprocessed_path", required=True)
def main(features_raw_path: str, features_preprocessed_path: str) -> None:
    features: np.ndarray = np.genfromtxt(features_raw_path, delimiter=',')

    os.makedirs(os.path.dirname(features_preprocessed_path), exist_ok=True)

    np.savetxt(features_preprocessed_path, features, delimiter=",")


if __name__ == '__main__':
    main()
