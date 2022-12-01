import os
import pickle
import click
import numpy as np

from sklearn.ensemble import RandomForestClassifier


@click.command()
@click.option("--features_path", required=True)
@click.option("--predictions_path", required=True)
@click.option("--model_path", required=True)
def main(features_path: str, predictions_path: str, model_path: str) -> None:
    features: np.ndarray
    features = np.genfromtxt(features_path, delimiter=',')

    with open(model_path, "rb") as file:
        model: RandomForestClassifier = pickle.load(file)

    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    np.savetxt(predictions_path, model.predict(features), delimiter=",")


if __name__ == '__main__':
    main()