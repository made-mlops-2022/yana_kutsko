import os
import click
import pickle
import pandas as pd
from loggers.logger import create_root_loger


@click.command(name='predict_model')
@click.argument('model_path')
@click.argument('data_csv_path')
@click.argument('output_path')
def train_model(model_path: str, data_csv_path: str, output_path: str):
    logger = create_root_loger()
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    x_pred = pd.read_csv(data_csv_path)
    y_pred = pd.Series(model.predict(x_pred))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_pred.to_csv(output_path)
    logger.info(f'Prediction is stored in {output_path}')
    print('Done')


if __name__ == "__main__":
    train_model()

