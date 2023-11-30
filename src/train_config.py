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


@dataclass
class Model:
    name: str


@dataclass
class ProtoNetTrain:
    epochs: int


@dataclass
class TrainConfig:
    dataset: Dataset
    model: Model
    protonet_train: ProtoNetTrain
