import click
import yaml
from model.model import Model
from parameters.train_model_params import TrainModelParams, TrainModelParamsSchema
from data.dataset_operations import DatasetOperations


def read_params(config_path: str) -> TrainModelParams:
    schema = TrainModelParamsSchema()
    with open(config_path, 'r') as config:
        return schema.load(yaml.safe_load(config))


@click.command('model')
@click.argument('config_path')
def train_model(config_path: str):
    print(config_path)
    config = read_params(config_path)
    dataset_operations = DatasetOperations()
    df = dataset_operations.read_dataset(config.data_params.path_to_data)
    features, target = dataset_operations.split_to_features_and_target(df)
    X_train, X_test, y_train, y_test = dataset_operations.split_to_train_and_test(features, target,
                                                                                  config.splitParams.test_size,
                                                                                  config.splitParams.random_state)
    model = Model(config.model_params.model_type)
    model.train(X_train, y_train)
    model.serialize_model(config.output_model_path)


if __name__ == '__main__':
    train_model('../../configs/knn.yaml')
