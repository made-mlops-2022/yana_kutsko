from dataclasses import dataclass, field


@dataclass
class ModelParams:
    model_type: str = field(default='KNeighborsClassifier')
