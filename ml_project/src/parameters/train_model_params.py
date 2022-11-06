from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from parameters.split_params import SplitParams
from parameters.features_params import FeaturesParams
from parameters.model_params import ModelParams
from parameters.data_params import DataParams


@dataclass
class TrainModelParams:
    output_model_path: str
    output_metrics_path: str
    data_params: DataParams
    features_params: FeaturesParams
    model_params: ModelParams
    splitParams: SplitParams


TrainModelParamsSchema = class_schema(TrainModelParams)
