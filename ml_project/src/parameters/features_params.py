from dataclasses import dataclass
from typing import List


@dataclass
class FeaturesParams:
    categorical_features: List[str]
    numeric_features: List[str]
    target_column: str
