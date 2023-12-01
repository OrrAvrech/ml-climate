import random
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Optional


random.seed(42)


class BaseImageDataset(Dataset):
    def __init__(
        self, root_dir: Path, classes: Optional[list] = None, transform: Optional = None
    ):
        self.root_dir = root_dir
        self.transform = transform
        if classes is None:
            self.classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for img_path in class_dir.iterdir():
                images.append((str(img_path), self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class BaseImageDatasetSplit(BaseImageDataset):
    TRAIN_SPLIT = 0.7

    def __init__(self, split: str, root_dir: Path, transform: Optional = None):
        train_split, val_split, test_split = split_classes(root_dir, self.TRAIN_SPLIT)
        cls_splits = {"train": train_split, "validation": val_split, "test": test_split}
        if split in cls_splits.keys():
            super().__init__(
                root_dir=root_dir, classes=cls_splits[split], transform=transform
            )
        else:
            raise ValueError("Splits needs to be train, test or validation")


class EuroSAT(BaseImageDatasetSplit):
    # EuroSAT RGB
    pass


class OPTIMAL31(BaseImageDatasetSplit):
    # OPTIMAL-31 General Scenes
    pass


def split_classes(root_dir: Path, train_split: float):
    test_split = (1 - train_split) / 2
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    num_cls = len(classes)
    train_num_cls = int(train_split * num_cls)
    val_num_cls = int(test_split * num_cls)

    train_cls_split = random.sample(classes, train_num_cls)
    remaining_list = [elem for elem in classes if elem not in train_cls_split]
    val_cls_split = random.sample(remaining_list, val_num_cls)
    test_cls_split = [elem for elem in remaining_list if elem not in val_cls_split]
    return train_cls_split, val_cls_split, test_cls_split
