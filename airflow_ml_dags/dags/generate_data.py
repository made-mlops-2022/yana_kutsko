import os
import random

import numpy as np
import pendulum

from sklearn.datasets import make_classification

from airflow import DAG
from airflow.operators.python import PythonOperator

from pathes import RAW_DATA_PATH, RAW_TARGET_PATH


def _synthesize_data(features_path: str, target_path: str) -> None:
    n_samples: int = random.randint(100, 1000)

    features: np.ndarray
    target: np.ndarray
    features, target = make_classification(n_samples=n_samples, n_features=10, n_informative=5)

    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    np.savetxt(features_path, features, delimiter=",")
    np.savetxt(target_path, target, delimiter=",")


with DAG(
        dag_id="generate_data",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        schedule_interval="0 0 * * *",
        tags=["ml_ops"]
) as dag:
    synthesize_data = PythonOperator(
        task_id="synthesize_data",
        python_callable=_synthesize_data,
        op_kwargs={
            "features_path": f"{RAW_DATA_PATH}",
            "target_path": f"{RAW_TARGET_PATH}",
        }
    )

    synthesize_data
