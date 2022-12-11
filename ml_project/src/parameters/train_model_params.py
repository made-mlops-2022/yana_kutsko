from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from ml_project.src.parameters.split_params import SplitParams
from ml_project.src.parameters.features_params import FeaturesParams
from ml_project.src.parameters.model_params import ModelParams
from ml_project.src.parameters.data_params import DataParams


@dataclass
class TrainModelParams:
    output_model_path: str
    output_metrics_path: str
    data_params: DataParams
    features_params: FeaturesParams
    model_params: ModelParams
    splitParams: SplitParams


TrainModelParamsSchema = class_schema(TrainModelParams)
