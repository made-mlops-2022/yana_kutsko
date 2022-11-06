import click
import pickle
import pandas as pd


@click.command(name='predict_model')
@click.argument('model_path')
@click.argument('data_csv_path')
@click.argument('output_path')
def train_model(model_path: str, data_csv_path: str, output_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    x_pred = pd.read_csv(data_csv_path)
    y_pred = pd.Series(model.predict(x_pred))
    y_pred.to_csv(output_path)


if __name__ == "__main__":
    train_model()

