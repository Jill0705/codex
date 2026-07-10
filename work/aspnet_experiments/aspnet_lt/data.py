import numpy as np
import torch
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


def build_dataset(name, data_root, imb_factor=100, seed=0, synthetic_size=1024, num_classes=None):
    name = name.lower()
    if name == 'synthetic':
        train = SyntheticLongTail(num_classes=num_classes or 10, size=synthetic_size, seed=seed)
        val = SyntheticLongTail(num_classes=num_classes or 10, size=max(256, synthetic_size // 4), seed=seed + 1)
        return train, val, train.num_classes, train.cls_num_list

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
