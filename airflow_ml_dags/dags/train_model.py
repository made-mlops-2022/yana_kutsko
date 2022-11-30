import os
import pickle
import json

import numpy as np
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from pathes import RAW_DATA_PATH, RAW_TARGET_PATH, PROCESSED_DATA_PATH, FEATURES_TRAIN, \
                   FEATURES_TEST, TARGET_TRAIN, TARGET_TEST, MODEL_PATH, METRICS_PATH


def _preprocess_data(features_raw_path: str, features_preprocessed_path: str) -> None:
    features: np.ndarray = np.genfromtxt(features_raw_path, delimiter=',')

    os.makedirs(os.path.dirname(features_preprocessed_path), exist_ok=True)

    np.savetxt(features_preprocessed_path, features, delimiter=",")


def _split_data(features_preprocessed_path: str,
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


def _train_model(features_train_path: str,
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


def _validate_model(features_test_path: str,
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


with DAG(
        dag_id="train_model",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        schedule_interval="@weekly",
        tags=["ml_ops"]
) as dag:
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_preprocess_data,
        op_kwargs={
            "features_raw_path": f"{RAW_DATA_PATH}",
            "features_preprocessed_path": f"{PROCESSED_DATA_PATH}",
        },
    )

    split_data = PythonOperator(
        task_id="split_data",
        python_callable=_split_data,
        op_kwargs={
            "features_preprocessed_path": f"{PROCESSED_DATA_PATH}",
            "target_path": f"{RAW_TARGET_PATH}",
            "features_train_path": f"{FEATURES_TRAIN}",
            "features_test_path": f"{FEATURES_TEST}",
            "target_train_path": f"{TARGET_TRAIN}",
            "target_test_path": f"{TARGET_TEST}",
        },
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        op_kwargs={
            "features_train_path": f"{FEATURES_TRAIN}",
            "target_train_path": f"{TARGET_TRAIN}",
            "model_path": f"{MODEL_PATH}",
        },
    )

    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=_validate_model,
        op_kwargs={
            "features_test_path": f"{FEATURES_TEST}",
            "target_test_path": f"{TARGET_TEST}",
            "model_path": f"{MODEL_PATH}",
            "metrics_path": f"{METRICS_PATH}",
        },
    )

    preprocess_data >> split_data >> train_model >> validate_model
