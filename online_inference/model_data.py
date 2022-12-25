from typing import Literal
from pydantic import BaseModel, validator


class ModelData(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: int
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator('age')
    def check_age(cls, v):
        if v < 0 or v > 125:
            raise ValueError('age is beyond allowed boundaries')
        return v

    @validator('trestbps')
    def check_trestbps(cls, v):
        if v < 0 or v > 200:
            raise ValueError('trestbps is beyond allowed boundaries.')
        return v

    @validator('thalach')
    def check_thalach(cls, v):
        if v < 0 or v > 250:
            raise ValueError('thalach is beyond allowed boundaries.')
        return v

    @validator('chol')
    def check_chol(cls, v):
        if v < 100 or v > 600:
            raise ValueError('chol is beyond allowed boundaries.')
        return v

    @validator('oldpeak')
    def check_oldpeak(cls, v):
        if v < 0 or v > 10:
            raise ValueError('oldpeak is beyond allowed boundaries.')
        return v