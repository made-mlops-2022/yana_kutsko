from dataclasses import dataclass, field


@dataclass
class SplitParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)
