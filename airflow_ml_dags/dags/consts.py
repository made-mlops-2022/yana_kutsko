from datetime import timedelta

from airflow.models import Variable
from airflow.utils.email import send_email_smtp

DATA_VOLUME_PATH = Variable.get('data_volume_path')

MODEL_PATH = Variable.get('model_path')

METRICS_PATH = Variable.get('metrics_path')

SAVE_MODEL_PATH = "/data/models/{{ ds }}/model.pkl"

RAW_DATA_PATH = "/data/raw/{{ ds }}/data.csv"

RAW_TARGET_PATH = "/data/raw/{{ ds }}/target.csv"

PROCESSED_DATA_PATH = "/data/processed/{{ ds }}/train_data.csv"

PREDICTIONS_PATH = "/data/predicted/{{ ds }}/predictions.csv"

FEATURES_TRAIN = "/data/split/{{ ds }}/features_train.csv"

FEATURES_TEST = "/data/split/{{ ds }}/features_test.csv"

TARGET_TRAIN = "/data/split/{{ ds }}/target_train.csv"

TARGET_TEST = "/data/split/{{ ds }}/target_test.csv"


def failure_email(context):
    dag_run = context.get('dag_run')
    task_instances = dag_run.get_task_instances()
    subject = f"These task instances failed: {task_instances}"
    send_email_smtp(to=default_args['email'], subject=subject)


default_args = {
    "owner": "yana_kutsko",
    "email": ['pasteyourtext.dev@gmail.com'],
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
    "on_failure_callback": failure_email
}

