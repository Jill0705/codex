import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float32)
        self.register_buffer('log_counts', torch.log(counts.clamp_min(1.0)))

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_counts.view(1, -1), target)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, scale=30.0, drw_epoch=0):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float32)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))
        margins = margins * (max_m / margins.max().clamp_min(1e-12))
        self.register_buffer('margins', margins)
        self.register_buffer('class_weights', torch.ones_like(counts))
        self.scale = float(scale)
        self.drw_epoch = int(drw_epoch)

        effective_num = 1.0 - torch.pow(torch.tensor(0.9999), counts)
        weights = (1.0 - 0.9999) / effective_num.clamp_min(1e-12)
        weights = weights / weights.sum().clamp_min(1e-12) * len(cls_num_list)
        self.register_buffer('deferred_weights', weights)

    def set_epoch(self, epoch):
        if self.drw_epoch > 0 and epoch >= self.drw_epoch:
            self.class_weights.copy_(self.deferred_weights)
        else:
            self.class_weights.fill_(1.0)

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_margins = self.margins[target].view(-1, 1)
        adjusted = torch.where(index, logits - batch_margins, logits)
        return F.cross_entropy(self.scale * adjusted, target, weight=self.class_weights)


def build_loss(name, cls_num_list, drw_epoch=0):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    if name == 'balanced_softmax':
        return BalancedSoftmaxLoss(cls_num_list)
    if name == 'ldam':
        return LDAMLoss(cls_num_list, drw_epoch=drw_epoch)
    raise ValueError(f'Unknown loss: {name}')
