import os
import json
import pickle

import click
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


@click.command()
@click.option("--features_test_path", required=True)
@click.option("--target_test_path", required=True)
@click.option("--model_path", required=True)
@click.option("--metrics_path", required=True)
def main(features_test_path: str,
         target_test_path: str,
         model_path: str,
         metrics_path: str) -> None:
    X_test: np.ndarray
    y_test: np.ndarray

    X_test = np.genfromtxt(features_test_path, delimiter=',')
    y_test = np.genfromtxt(target_test_path, delimiter=',')

    with open(model_path, "rb") as file:
        model: RandomForestClassifier = pickle.load(file)

        y_pred = model.predict(X_test)

        score = {'f1_score': f1_score(y_test, y_pred),
                 'roc_auc_score': roc_auc_score(y_test, y_pred),
                 'accuracy_score': accuracy_score(y_test, y_pred)}

        with open(metrics_path, 'w+') as file_metrics:
            json.dump(score, file_metrics)


if __name__ == '__main__':
    main()
