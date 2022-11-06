from dataclasses import dataclass, field


@dataclass
class DataParams:
    path_to_data: str = field(default='../../data/heart_cleveland.csv')
