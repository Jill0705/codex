import csv
import json
import os
import random
from dataclasses import asdict, is_dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += int(n)


def accuracy(logits, target):
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_csv(path, row):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def classwise_accuracy(logits, targets, num_classes):
    pred = logits.argmax(dim=1)
    correct = torch.zeros(num_classes, dtype=torch.float64)
    total = torch.zeros(num_classes, dtype=torch.float64)
    for c in range(num_classes):
        mask = targets == c
        total[c] = mask.sum().item()
        if total[c] > 0:
            correct[c] = (pred[mask] == c).sum().item()
    acc = torch.where(total > 0, correct / total * 100.0, torch.zeros_like(total))
    return acc, total


def split_class_groups(train_counts):
    counts = np.asarray(train_counts)
    order = np.argsort(-counts)
    n = len(counts)
    many = set(order[: n // 3].tolist())
    medium = set(order[n // 3 : 2 * n // 3].tolist())
    few = set(order[2 * n // 3 :].tolist())
    return many, medium, few


def group_accuracy(class_acc, train_counts):
    many, medium, few = split_class_groups(train_counts)
    values = class_acc.detach().cpu().numpy()

    def mean_for(group):
        idx = sorted(group)
        return float(values[idx].mean()) if idx else 0.0

    return {
        'many_acc': mean_for(many),
        'medium_acc': mean_for(medium),
        'few_acc': mean_for(few),
        'balanced_acc': float(values.mean()),
    }
