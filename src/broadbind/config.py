from pydantic import BaseModel
from pathlib import Path


class Model(BaseModel):
    hidden_channels: int
    dropout_rate: float
    hidden_dimension: int
    k_neighbors: int


class Training(BaseModel):
    batch_size: int
    max_epoch: int
    learning_rate: float


class Config(BaseModel):
    bound_sites: list[str]
    properties: list[str]
    path: Path
    training: Training
    model: Model
