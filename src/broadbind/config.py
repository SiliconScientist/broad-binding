from pydantic import BaseModel
from pathlib import Path


class Model(BaseModel):
    hidden_channels: int
    hidden_dimension: int
    k_neighbors: int


class Training(BaseModel):
    batch_size: int
    max_epoch: int
    learning_rate: float
    decay_period: int
    gradient_clip: float
    max_epochs: int


class Paths(BaseModel):
    train: Path
    validation: Path
    test: Path
    checkpoints: Path
    logs: Path


class Logging(BaseModel):
    checkpoint_every_n_steps: int
    log_every_n_steps: int


class Config(BaseModel):
    random_seed: int
    bound_sites: list[str]
    properties: list[str]
    training: Training
    model: Model
    paths: Paths
    logging: Logging
    fast_dev_run: bool
