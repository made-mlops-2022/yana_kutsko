from airflow.models import Variable


DATA_VOLUME_PATH = Variable.get('data_volume_path')

MODEL_PATH = Variable.get('model_path')

RAW_DATA_PATH = "/data/raw/{{ ds }}/data.csv"

RAW_TARGET_PATH = "/data/raw/{{ ds }}/target.csv"

PROCESSED_DATA_PATH = "/data/processed/{{ ds }}/train_data.csv"

PREDICTIONS_PATH = "/data/predicted/{{ ds }}/predictions.csv"

FEATURES_TRAIN = "/data/split/{{ ds }}/features_train.csv"

FEATURES_TEST = "/data/split/{{ ds }}/features_test.csv"

TARGET_TRAIN = "/data/split/{{ ds }}/target_train.csv"

TARGET_TEST = "/data/split/{{ ds }}/target_test.csv"

METRICS_PATH = "/data/model/{{ ds }}/metrics.json"

