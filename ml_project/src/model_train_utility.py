import click
from ml_project.src.model.model import Model
from ml_project.src.data.dataset_operations import DatasetOperations
from ml_project.src.parameters.read_params import read_params
from ml_project.src.loggers.logger import create_root_loger


def train_model(config_path):
    logger = create_root_loger()
    config = read_params(config_path)
    dataset_operations = DatasetOperations()
    df = dataset_operations.read_dataset(config.data_params.path_to_data)
    features, target = dataset_operations.split_to_features_and_target(df)
    X_train, X_test, y_train, y_test = dataset_operations.split_to_train_and_test(features, target,
                                                                                  config.splitParams.test_size,
                                                                                  config.splitParams.random_state)
    model = Model(config.model_params.model_type,
                  config.features_params.categorical_features,
                  config.features_params.numerical_features)
    model.train(X_train, y_train)
    model.serialize_model(config.output_model_path)
    logger.info(f'Model in stored in {config.output_model_path}')


@click.command(name='train_model_command')
@click.argument('config_path')
def train_model_command(config_path: str):
    train_model(config_path)
    print('Done')


if __name__ == "__main__":
    train_model_command()

