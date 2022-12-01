import os
import pickle
import json

import numpy as np
import pendulum

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from pathes import DATA_VOLUME_PATH, MODEL_PATH, RAW_DATA_PATH, PREDICTIONS_PATH


with DAG(
        dag_id="make_predictions",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        schedule_interval="@daily",
        tags=["ml_ops"]
) as dag:

    predict = DockerOperator(
        image="predict",
        command=f"--features_path {RAW_DATA_PATH} "
                f"--predictions_path {PREDICTIONS_PATH} "
                f"--model_path {MODEL_PATH}",
        task_id="docker-airflow-predict",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    predict
