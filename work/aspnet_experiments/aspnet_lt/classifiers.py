import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize_weights(weights):
    weights = weights.float().clamp_min(0.0)
    if float(weights.sum()) <= 0.0:
        return torch.ones_like(weights) / max(1, weights.numel())
    return weights / weights.sum()


def _largest_remainder_counts(weights, target_budget, k_min, k_max):
    num_classes = int(weights.numel())
    min_budget = num_classes * k_min
    max_budget = num_classes * k_max
    target_budget = int(max(min_budget, min(max_budget, target_budget)))
    capacity = torch.full((num_classes,), k_max - k_min, dtype=torch.float32)
    remaining = float(target_budget - min_budget)

    if remaining <= 0:
        return [k_min] * num_classes
    if remaining >= float(capacity.sum()):
        return [k_max] * num_classes

    weights = _safe_normalize_weights(weights)
    extra = torch.zeros(num_classes, dtype=torch.float32)
    active = capacity > 0
    remaining_float = remaining

    while bool(active.any()):
        active_idx = active.nonzero(as_tuple=False).flatten()
        active_weights = _safe_normalize_weights(weights[active_idx])
        proposal = remaining_float * active_weights
        active_capacity = capacity[active_idx]
        over = proposal >= active_capacity
        if not bool(over.any()):
            extra[active_idx] = proposal
            break

        capped_idx = active_idx[over]
        extra[capped_idx] = capacity[capped_idx]
        remaining_float -= float(capacity[capped_idx].sum())
        active[capped_idx] = False

    floors = torch.floor(extra).long()
    counts = torch.full((num_classes,), k_min, dtype=torch.long) + floors
    leftover = target_budget - int(counts.sum().item())
    if leftover > 0:
        remainders = extra - floors.float()
        can_add = counts < k_max
        remainders = torch.where(can_add, remainders, torch.full_like(remainders, -1.0))
        order = torch.argsort(remainders, descending=True)
        added = 0
        for idx in order.tolist():
            if added >= leftover:
                break
            if counts[idx] < k_max:
                counts[idx] += 1
                added += 1
    return counts.tolist()


def _allocation_weights(cls_num_list, allocation='log', effective_beta=0.9999):
    counts = torch.tensor(cls_num_list, dtype=torch.float32)
    allocation = allocation.lower()
    if allocation == 'linear':
        return counts.clamp_min(1.0)
    if allocation == 'sqrt':
        return torch.sqrt(counts.clamp_min(1.0))
    if allocation == 'log':
        return torch.log(counts.clamp_min(1.0))
    if allocation == 'effective':
        beta = float(effective_beta)
        return (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts.clamp_min(1.0))) / max(1e-12, 1.0 - beta)
    raise ValueError(f'Unknown allocation: {allocation}')


def adaptive_proto_counts(cls_num_list, k_max=4, k_min=1, allocation='log', proto_budget=None, effective_beta=0.9999):
    counts = torch.tensor(cls_num_list, dtype=torch.float32)
    if counts.numel() == 1:
        return [int(k_max)]

    weights = _allocation_weights(cls_num_list, allocation=allocation, effective_beta=effective_beta)
    if proto_budget is not None:
        return _largest_remainder_counts(weights, int(proto_budget), int(k_min), int(k_max))

    if float(weights.max()) == float(weights.min()):
        return [int(k_max)] * int(weights.numel())
    ratio = (weights - weights.min()) / (weights.max() - weights.min()).clamp_min(1e-6)
    k = int(k_min) + torch.round(ratio * (int(k_max) - int(k_min))).long()
    return k.clamp(1, k_max).tolist()


class LinearHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


class SubPrototypeHead(nn.Module):
    def __init__(self, feature_dim, num_classes, proto_counts, temperature=0.1, pooling='lse', pool_tau=0.25):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.proto_counts = [int(k) for k in proto_counts]
        self.temperature = float(temperature)
        self.pooling = pooling
        self.pool_tau = float(pool_tau)
        self.memory_update = 'grad'
        self.ema_momentum = 0.9
        self.confidence_temp = 4.0
        self.offsets = []
        class_ids = []
        local_ids = []
        start = 0
        for class_idx, k in enumerate(self.proto_counts):
            self.offsets.append((start, start + k))
            class_ids.extend([class_idx] * k)
            local_ids.extend(list(range(k)))
            start += k
        self.prototypes = nn.Parameter(torch.randn(start, feature_dim) * 0.02)
        self.max_proto_count = max(self.proto_counts)
        mask = torch.zeros(num_classes, self.max_proto_count, dtype=torch.bool)
        for class_idx, k in enumerate(self.proto_counts):
            mask[class_idx, :k] = True
        self.register_buffer('proto_mask', mask)
        self.register_buffer('proto_class_ids', torch.tensor(class_ids, dtype=torch.long))
        self.register_buffer('proto_local_ids', torch.tensor(local_ids, dtype=torch.long))

    def configure_memory_update(self, mode='grad', ema_momentum=0.9, confidence_temp=4.0):
        self.memory_update = mode
        self.ema_momentum = float(ema_momentum)
        self.confidence_temp = float(confidence_temp)
        if mode != 'grad':
            self.prototypes.requires_grad_(False)

    @classmethod
    def single(cls, feature_dim, num_classes, **kwargs):
        return cls(feature_dim, num_classes, [1] * num_classes, **kwargs)

    @classmethod
    def fixed(cls, feature_dim, num_classes, fixed_k=4, **kwargs):
        return cls(feature_dim, num_classes, [fixed_k] * num_classes, **kwargs)

    @classmethod
    def adaptive(cls, feature_dim, num_classes, cls_num_list, k_max=4, k_min=1, allocation='log', proto_budget=None, effective_beta=0.9999, **kwargs):
        return cls(
            feature_dim,
            num_classes,
            adaptive_proto_counts(
                cls_num_list,
                k_max=k_max,
                k_min=k_min,
                allocation=allocation,
                proto_budget=proto_budget,
                effective_beta=effective_beta,
            ),
            **kwargs,
        )

    def forward(self, features):
        z = F.normalize(features, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        padded = p.new_zeros(self.num_classes, self.max_proto_count, self.feature_dim)
        padded[self.proto_class_ids, self.proto_local_ids] = p
        sims = torch.einsum('bd,ckd->bck', z, padded) / self.temperature
        mask = self.proto_mask.unsqueeze(0)

        if self.pooling == 'max':
            return sims.masked_fill(~mask, torch.finfo(sims.dtype).min).max(dim=2).values
        if self.pooling == 'mean':
            valid_sims = sims.masked_fill(~mask, 0.0)
            denom = self.proto_mask.sum(dim=1).clamp_min(1).to(sims.dtype).view(1, -1)
            return valid_sims.sum(dim=2) / denom
        if self.pooling == 'lse':
            masked = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
            return self.pool_tau * torch.logsumexp(masked / self.pool_tau, dim=2)
        raise ValueError(f'Unknown pooling: {self.pooling}')

    @torch.no_grad()
    def update_memory(self, features, targets):
        if self.memory_update == 'grad':
            return

        z = F.normalize(features.detach(), dim=1)
        prototypes = F.normalize(self.prototypes.data, dim=1)
        base_rate = 1.0 - self.ema_momentum

        for class_idx, (start, end) in enumerate(self.offsets):
            mask = targets == class_idx
            if not torch.any(mask):
                continue

            class_features = z[mask]
            class_protos = prototypes[start:end]
            assignments = (class_features @ class_protos.t()).argmax(dim=1)

            for local_idx in range(end - start):
                proto_mask = assignments == local_idx
                count = int(proto_mask.sum().item())
                if count == 0:
                    continue

                center = F.normalize(class_features[proto_mask].mean(dim=0), dim=0)
                if self.memory_update == 'conf_ema':
                    confidence = count / (count + self.confidence_temp)
                else:
                    confidence = 1.0
                rate = base_rate * confidence
                global_idx = start + local_idx
                updated = (1.0 - rate) * self.prototypes.data[global_idx] + rate * center
                self.prototypes.data[global_idx] = F.normalize(updated, dim=0)


class RecognitionModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
