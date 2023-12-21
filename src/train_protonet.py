import pyrallis
import numpy as np
from typing import Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import transformers.data.metrics.squad_metrics
from torch.utils.data import DataLoader, Dataset

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from torchvision.transforms import Resize, ToTensor, Normalize
from train_config import TrainConfig
from data.datasets import OPTIMAL31
from models.base import BasePreTrained
from models.scale_mae import mae_vit_large_patch16


class OPTIMAL_DATASET_STATS:
    # Divided by 255
    PIXEL_MEANS = [
        0.3688422134901961,
        0.3807842425882353,
        0.3377770719607843,
    ]  # [0.485, 0.456, 0.406] # imagenet
    PIXEL_STD = [
        0.20127227078431373,
        0.1805812771764706,
        0.1775864696862745,
    ]  # [0.229, 0.224, 0.225]  # imagenet


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

    eval_scale = 256
    eval_res = 1.0

    data = data.to(torch.float32)
    data = Normalize(mean=OPTIMAL_DATASET_STATS.PIXEL_MEANS, std=OPTIMAL_DATASET_STATS.PIXEL_STD)(data)

    data = torch.nn.functional.interpolate(
        data, (eval_scale, eval_scale), mode="area"
    )
    input_res = torch.ones(len(data)).float().to(data.device) * eval_res

    embeddings = model(data, knn_feats=True, input_res=input_res)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

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
        shuffle=False,
    )

    test_loader = get_data_loader(
        test_dataset,
        meta=True,
        num_tasks=cfg.dataset.test_tasks,
        way=cfg.dataset.test_way,
        shot=cfg.dataset.test_shot,
        query=cfg.dataset.test_query,
        shuffle=False,
    )

    # model = BasePreTrained(cfg.model.name, device=device)
    weights_path = "./weights/scalemae-vitlarge-800.pth"
    model = mae_vit_large_patch16(fixed_output_size=0, independent_fcn_head=True, fcn_dim=512,
                                  fcn_layers=2, decoder_depth=3, absolute_scale=True)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["model"])
    model.to(device)

    if cfg.model.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=cfg.dataset.train_tasks, eta_min=cfg.training.min_lr
        # )

        best_val_loss = float("inf")
        stop_counter = 0

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()

            loss_ctr, n_loss, n_acc = 0, 0, 0

            for i in range(cfg.dataset.train_tasks):
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

                if i % 10 == 0:
                    print(
                        f"epoch {epoch} - task {i}, train, loss={loss.item():.4f} acc={acc:.4f}"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # lr_scheduler.step()

            train_loss = n_loss / loss_ctr
            train_acc = n_acc / loss_ctr
            print(f"epoch {epoch}, train, loss={train_loss:.4f} acc={train_acc:.4f}")

            model.eval()
            loss_ctr, n_loss, n_acc = 0, 0, 0
            for i, batch in enumerate(val_loader):
                with torch.no_grad():
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

            val_loss = n_loss / loss_ctr
            val_acc = n_acc / loss_ctr
            print(f"epoch {epoch}, val, loss={val_loss:.4f} acc={val_acc:.4f}")

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stop_counter = 0  # Reset counter if there's improvement
                suffix = datetime.now().strftime("%Y%m%d_%H%M")
                save_path = cfg.training.saved_models_dir / f"protonet_{suffix}.pt"
                torch.save(model.state_dict(), save_path)
            else:
                stop_counter += 1
                if stop_counter >= cfg.training.patience:
                    print(
                        f"Early stopping at epoch {epoch}. Validation loss did not improve."
                    )
                    break  # Stop training

    for i, batch in enumerate(test_loader, 1):
        with torch.no_grad():
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
        print(f"batch {i}: {n_acc / loss_ctr * 100:.2f}({acc * 100:.2f})")


if __name__ == "__main__":
    main()
