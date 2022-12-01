import pendulum

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from docker.types import Mount
from airflow.models import Variable

from pathes import RAW_DATA_PATH, RAW_TARGET_PATH, PROCESSED_DATA_PATH, FEATURES_TRAIN, \
    FEATURES_TEST, TARGET_TRAIN, TARGET_TEST, MODEL_PATH, METRICS_PATH, DATA_VOLUME_PATH, SAVE_MODEL_PATH


def _set_model_variable(path: str) -> None:
    Variable.set('model_path', path)


with DAG(
        dag_id="train_model",
        start_date=pendulum.datetime(2022, 11, 28, tz="UTC"),
        schedule_interval="@weekly",
        tags=["ml_ops"]
) as dag:
    preprocess_data = DockerOperator(
        image="preprocess-data",
        command=f"--features_raw_path {RAW_DATA_PATH} "
                f"--features_preprocessed_path {PROCESSED_DATA_PATH}",
        task_id="preprocess",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    split_data = DockerOperator(
        image="split-data",
        command=f"--features_preprocessed_path {PROCESSED_DATA_PATH} "
                f"--target_path {RAW_TARGET_PATH} "
                f"--features_train_path {FEATURES_TRAIN} "
                f"--features_test_path {FEATURES_TEST} "
                f"--target_train_path {TARGET_TRAIN} "
                f"--target_test_path {TARGET_TEST} ",
        task_id="split",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    train_model = DockerOperator(
        image="train-model",
        command=f"--features_train_path {FEATURES_TRAIN} "
                f"--target_train_path {TARGET_TRAIN} "
                f"--model_path {SAVE_MODEL_PATH}",
        task_id="train",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    set_model_path = PythonOperator(
        task_id="set_model_path",
        python_callable=_set_model_variable,
        op_kwargs={
            "path": f"{SAVE_MODEL_PATH}"
        }
    )

    validate_model = DockerOperator(
        image="validate-model",
        command=f"--features_test_path {FEATURES_TEST} "
                f"--target_test_path {TARGET_TEST} "
                f"--model_path {SAVE_MODEL_PATH} "
                f"--metrics_path {METRICS_PATH}",
        task_id="validate",
        network_mode='host',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=DATA_VOLUME_PATH, target='/data', type='bind')]
    )

    preprocess_data >> split_data >> train_model >> set_model_path >> validate_model
