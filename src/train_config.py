from dataclasses import dataclass


@dataclass
class Dataset:
    shot: int
    test_way: int
    test_shot: int
    test_query: int
    train_query: int
    train_way: int


@dataclass
class ProtoNetTrain:
    epochs: int


@dataclass
class TrainConfig:
    dataset: Dataset
    protonet_train: ProtoNetTrain
