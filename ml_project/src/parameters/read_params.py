import yaml
from parameters.train_model_params import TrainModelParams, TrainModelParamsSchema


def read_params(config_path: str) -> TrainModelParams:
    schema = TrainModelParamsSchema()
    with open(config_path, 'r') as config:
        return schema.load(yaml.safe_load(config))
