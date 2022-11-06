from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from split_param import SplitParams
from features_params import FeaturesParams
from model_params import ModelParams
from data_params import DataParams


@dataclass
class TrainModelParams:
    output_model_path: str
    output_metrics_path: str
    data_params: DataParams
    features_params: FeaturesParams
    model_params: ModelParams
    splitParams: SplitParams


TrainModelParamsSchema = class_schema(TrainModelParams)
