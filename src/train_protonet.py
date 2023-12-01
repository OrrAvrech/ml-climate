import pyrallis
import numpy as np
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from train_config import TrainConfig
from data.datasets import OPTIMAL31
from models.vit import ViTPreTrained

from transformers import ViTModel, AutoImageProcessor


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -(
        (a.unsqueeze(1).expand(n, m, -1) - b.unsqueeze(0).expand(n, m, -1)) ** 2
    ).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    data = data.to(device)
    labels = labels.to(device)
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def get_data_loader(
    dataset: Dataset,
    meta: bool,
    num_tasks: int = 1,
    shuffle: bool = True,
    way: Optional[int] = None,
    shot: Optional[int] = None,
    query: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    if meta is True:
        ds = l2l.data.MetaDataset(dataset)
        transforms = [
            NWays(ds, way),
            KShots(ds, query + shot),
            LoadData(ds),
            RemapLabels(ds),
        ]
        tasks = l2l.data.Taskset(ds, task_transforms=transforms, num_tasks=num_tasks)
        data_loader = DataLoader(tasks, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


@pyrallis.wrap()
def main(cfg: TrainConfig):
    loss_ctr = 0
    n_acc = 0
    device = torch.device("cpu")
    if torch.cuda.device_count():
        torch.cuda.manual_seed(43)
        device = torch.device("cuda")
    print(device)

    model = ViTPreTrained(cfg.model.name, device=device)
    # processor = AutoImageProcessor.from_pretrained(cfg.model.name)
    # model = ViTModel.from_pretrained(cfg.model.name)
    model = model.to(device)

    train_dataset = OPTIMAL31(root_dir=cfg.dataset.root_dir, split="train")
    val_dataset = OPTIMAL31(root_dir=cfg.dataset.root_dir, split="validation")
    test_dataset = OPTIMAL31(root_dir=cfg.dataset.root_dir, split="test")

    train_loader = get_data_loader(
        train_dataset,
        meta=cfg.training.meta,
        way=cfg.dataset.train_way,
        shot=cfg.dataset.shot,
        query=cfg.dataset.train_query,
    )

    val_loader = get_data_loader(
        val_dataset,
        meta=True,
        num_tasks=cfg.dataset.test_tasks,
        way=cfg.dataset.test_way,
        shot=cfg.dataset.test_shot,
        query=cfg.dataset.test_query,
    )

    test_loader = get_data_loader(
        test_dataset,
        meta=True,
        num_tasks=cfg.dataset.test_tasks,
        way=cfg.dataset.test_way,
        shot=cfg.dataset.test_shot,
        query=cfg.dataset.test_query,
    )

    if cfg.model.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()

            loss_ctr = 0
            n_loss = 0
            n_acc = 0

            for i in range(100):
                batch = next(iter(train_loader))

                loss, acc = fast_adapt(
                    model,
                    batch,
                    cfg.dataset.train_way,
                    cfg.dataset.shot,
                    cfg.dataset.train_query,
                    metric=pairwise_distances_logits,
                    device=device,
                )

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            print(
                "epoch {}, train, loss={:.4f} acc={:.4f}".format(
                    epoch, n_loss / loss_ctr, n_acc / loss_ctr
                )
            )

            model.eval()

            loss_ctr = 0
            n_loss = 0
            n_acc = 0
            for i, batch in enumerate(val_loader):
                loss, acc = fast_adapt(
                    model,
                    batch,
                    cfg.dataset.test_way,
                    cfg.dataset.test_shot,
                    cfg.dataset.test_query,
                    metric=pairwise_distances_logits,
                    device=device,
                )

                loss_ctr += 1
                n_loss += loss.item()
                n_acc += acc

            print(
                "epoch {}, val, loss={:.4f} acc={:.4f}".format(
                    epoch, n_loss / loss_ctr, n_acc / loss_ctr
                )
            )

    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(
            model,
            batch,
            cfg.dataset.test_way,
            cfg.dataset.test_shot,
            cfg.dataset.test_query,
            metric=pairwise_distances_logits,
            device=device,
        )
        loss_ctr += 1
        n_acc += acc
        print("batch {}: {:.2f}({:.2f})".format(i, n_acc / loss_ctr * 100, acc * 100))


if __name__ == "__main__":
    main()
