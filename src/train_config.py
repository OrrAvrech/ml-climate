from pathlib import Path
from dataclasses import dataclass


@dataclass
class Dataset:
    root_dir: Path
    shot: int
    test_way: int
    test_shot: int
    test_query: int
    train_query: int
    train_way: int
    train_tasks: int
    test_tasks: int


@dataclass
class Model:
    name: str
    train: bool


@dataclass
class Training:
    meta: bool
    epochs: int
    batch_size: int
    lr: float
    min_lr: float
    patience: int
    saved_models_dir: Path

    def __post_init__(self):
        self.saved_models_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class TrainConfig:
    dataset: Dataset
    model: Model
    training: Training
