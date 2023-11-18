import pyrallis
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from train_config import TrainConfig


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


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            hidden_size=hid_dim,
            channels=x_dim,
            max_pool=True,
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
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


@pyrallis.wrap()
def main(cfg: TrainConfig):
    loss_ctr = 0
    n_acc = 0
    device = torch.device("cpu")
    if torch.cuda.device_count():
        torch.cuda.manual_seed(43)
        device = torch.device("cuda")
    print(device)

    model = Convnet()
    model.to(device)

    path_data = "~/data"
    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode="train", download=True
    )
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode="validation", download=True
    )
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode="test", download=True
    )

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, cfg.dataset.train_way),
        KShots(train_dataset, cfg.dataset.train_query + cfg.dataset.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, cfg.dataset.test_way),
        KShots(valid_dataset, cfg.dataset.test_query + cfg.dataset.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.Taskset(
        valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=200,
    )
    valid_loader = DataLoader(valid_tasks, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, cfg.dataset.test_way),
        KShots(test_dataset, cfg.dataset.test_query + cfg.dataset.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=2000,
    )
    test_loader = DataLoader(test_tasks, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, cfg.protonet_train.epochs + 1):
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
        for i, batch in enumerate(valid_loader):
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
