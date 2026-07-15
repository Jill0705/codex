import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class SyntheticLongTail(Dataset):
    def __init__(self, num_classes=10, size=1024, image_size=32, seed=0):
        rng = np.random.default_rng(seed)
        self.x = torch.randn(size, 3, image_size, image_size)
        labels = np.arange(size) % num_classes
        rng.shuffle(labels)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.num_classes = num_classes
        self.cls_num_list = [int((self.y == c).sum().item()) for c in range(num_classes)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ImageListDataset(Dataset):
    def __init__(self, image_root, list_path, transform=None):
        if not image_root:
            raise ValueError('image_root is required for list-file datasets')
        if not list_path:
            raise ValueError('list_path is required for list-file datasets')
        self.image_root = image_root
        self.transform = transform
        self.samples = self._read_list(list_path)
        self.targets = [label for _, label in self.samples]

    @staticmethod
    def _read_list(list_path):
        samples = []
        with open(list_path, 'r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    rel_path, label = line.rsplit(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(f'Invalid list line {line_no} in {list_path}: {raw!r}') from exc
                samples.append((rel_path, int(label)))
        if not samples:
            raise ValueError(f'No samples found in {list_path}')
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rel_path, label = self.samples[index]
        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.image_root, rel_path)
        with Image.open(path) as img:
            image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_imbalance_counts(max_count, num_classes, imb_factor):
    if imb_factor <= 1:
        return [max_count] * num_classes
    counts = []
    for cls_idx in range(num_classes):
        exponent = cls_idx / max(1, num_classes - 1)
        counts.append(int(max_count * (1.0 / imb_factor) ** exponent))
    return counts


def build_long_tailed_subset(dataset, num_classes, imb_factor, seed=0):
    targets = np.array(dataset.targets)
    rng = np.random.default_rng(seed)
    max_count = min([(targets == c).sum() for c in range(num_classes)])
    cls_num_list = make_imbalance_counts(max_count, num_classes, imb_factor)
    selected = []
    for c, n in enumerate(cls_num_list):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[:n].tolist())
    rng.shuffle(selected)
    return Subset(dataset, selected), cls_num_list


def class_counts_from_targets(targets, num_classes=None):
    if num_classes is None:
        num_classes = int(max(targets)) + 1
    counts = [0] * int(num_classes)
    for label in targets:
        if label < 0 or label >= num_classes:
            raise ValueError(f'Label {label} is outside [0, {num_classes})')
        counts[int(label)] += 1
    return counts


def build_list_dataset(name, image_root, train_list, val_list, image_size=224, num_classes=None):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_resize = int(round(image_size * 256 / 224))
    val_tf = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train = ImageListDataset(image_root, train_list, transform=train_tf)
    val = ImageListDataset(image_root, val_list, transform=val_tf)
    inferred = max(max(train.targets), max(val.targets)) + 1
    num_classes = int(num_classes or inferred)
    cls_num_list = class_counts_from_targets(train.targets, num_classes=num_classes)
    class_counts_from_targets(val.targets, num_classes=num_classes)
    return train, val, num_classes, cls_num_list


def build_dataset(
    name,
    data_root,
    imb_factor=100,
    seed=0,
    synthetic_size=1024,
    num_classes=None,
    image_root=None,
    train_list=None,
    val_list=None,
    image_size=224,
):
    name = name.lower()
    if name == 'synthetic':
        train = SyntheticLongTail(num_classes=num_classes or 10, size=synthetic_size, seed=seed)
        val = SyntheticLongTail(num_classes=num_classes or 10, size=max(256, synthetic_size // 4), seed=seed + 1)
        return train, val, train.num_classes, train.cls_num_list

    if name in {'imagenetlt', 'placeslt', 'inatlt'}:
        return build_list_dataset(
            name,
            image_root=image_root or data_root,
            train_list=train_list,
            val_list=val_list,
            image_size=image_size,
            num_classes=num_classes,
        )

    if name not in {'cifar10lt', 'cifar100lt'}:
        raise ValueError(f'Unknown dataset: {name}')

    is_cifar100 = name == 'cifar100lt'
    num_classes = 100 if is_cifar100 else 10
    mean = (0.5071, 0.4867, 0.4408) if is_cifar100 else (0.4914, 0.4822, 0.4465)
    std = (0.2675, 0.2565, 0.2761) if is_cifar100 else (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_cls = datasets.CIFAR100 if is_cifar100 else datasets.CIFAR10
    full_train = ds_cls(root=data_root, train=True, transform=train_tf, download=True)
    test = ds_cls(root=data_root, train=False, transform=test_tf, download=True)
    train, cls_num_list = build_long_tailed_subset(full_train, num_classes, imb_factor, seed)
    return train, test, num_classes, cls_num_list
