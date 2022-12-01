import pendulum

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from pathes import RAW_DATA_PATH, RAW_TARGET_PATH, DATA_VOLUME_PATH


with DAG(
        dag_id="generate_data",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        schedule_interval="@daily",
        tags=["ml_ops"]
) as dag:
    predict = DockerOperator(
        image="generate-data",
        command=f"--features_path {RAW_DATA_PATH} "
                f"--target_path {RAW_TARGET_PATH}",
        task_id="generate_data",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    predict
