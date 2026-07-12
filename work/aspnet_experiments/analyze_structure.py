import argparse
import csv
import json
import os
import warnings
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from aspnet_lt.data import build_dataset
from aspnet_lt.utils import ensure_dir, split_class_groups
from train import build_model


DEFAULT_ARGS = {
    'allocation': 'log',
    'confidence_temp': 4.0,
    'data_root': 'data',
    'dataset': 'cifar100lt',
    'device': 'cuda',
    'effective_beta': 0.9999,
    'ema_momentum': 0.9,
    'feature_dim': 64,
    'fixed_k': 4,
    'imb_factor': 100,
    'k_max': 4,
    'k_min': 1,
    'loss': 'ce',
    'model': 'ce',
    'num_classes': None,
    'pool_tau': 0.25,
    'pooling': 'lse',
    'proto_budget': None,
    'proto_mode': 'single',
    'prototype_update': 'grad',
    'seed': 1,
    'synthetic_size': 1024,
    'temperature': 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze sample-supported intra-class structure.')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--output-dir', default='tables')
    parser.add_argument('--figure-dir', default='figures')
    parser.add_argument('--output-prefix', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-clusters', type=int, default=8)
    parser.add_argument('--min-cluster-samples', type=int, default=12)
    parser.add_argument('--bootstrap-iters', type=int, default=5)
    parser.add_argument('--bootstrap-size', type=int, default=160)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def load_training_args(run_dir):
    args_path = os.path.join(run_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r', encoding='utf-8') as f:
            args = json.load(f)
    else:
        checkpoint = torch.load(os.path.join(run_dir, 'best.pt'), map_location='cpu', weights_only=False)
        args = checkpoint.get('args', {})
    merged = dict(DEFAULT_ARGS)
    merged.update(args)
    return SimpleNamespace(**merged)


def deterministic_train_transform(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar100lt':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name == 'cifar10lt':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        return None
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def set_subset_transform(dataset, transform):
    if transform is None:
        return
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, 'transform'):
        base.transform = transform


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    targets = []
    for images, labels in tqdm(loader, desc='extract', leave=False):
        images = images.to(device, non_blocking=True)
        feat = model.backbone(images)
        feat = F.normalize(feat, dim=1)
        features.append(feat.cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def intra_class_variance(x):
    if len(x) <= 1:
        return 0.0
    center = x.mean(axis=0, keepdims=True)
    return float(np.mean(np.sum((x - center) ** 2, axis=1)))


def estimate_cluster_count(x, max_clusters, min_samples, seed):
    n = len(x)
    if n < max(min_samples, 3):
        return 1, np.nan
    max_k = min(max_clusters, n - 1)
    best_k = 1
    best_score = -1.0
    for k in range(2, max_k + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                labels = KMeans(n_clusters=k, random_state=seed, n_init=5, max_iter=100).fit_predict(x)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(x, labels, metric='euclidean'))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
    if best_score < 0.0:
        return 1, np.nan
    return best_k, best_score


def bootstrap_cluster_stability(x, max_clusters, min_samples, iters, sample_size, seed):
    if iters <= 0 or len(x) < max(min_samples, 3):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ks = []
    draw_size = min(len(x), sample_size)
    if draw_size < max(min_samples, 3):
        return np.nan, np.nan
    for _ in range(iters):
        idx = rng.choice(len(x), size=draw_size, replace=False)
        k, _ = estimate_cluster_count(x[idx], max_clusters, min_samples, int(rng.integers(1_000_000)))
        ks.append(k)
    counts = np.bincount(np.asarray(ks, dtype=np.int64))
    return float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0, float(counts.max() / len(ks))


def pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def rankdata(values):
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_groups(rows):
    out = []
    metrics = ['sample_count', 'intra_variance', 'estimated_k', 'best_silhouette', 'bootstrap_k_std', 'bootstrap_mode_fraction']
    for group in ['many', 'medium', 'few']:
        items = [r for r in rows if r['group'] == group]
        row = {'group': group, 'num_classes': len(items)}
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[f'{metric}_mean'] = float(vals.mean()) if len(vals) else np.nan
            row[f'{metric}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def build_stability_rows(rows):
    out = []
    for group in ['many', 'medium', 'few']:
        items = [r for r in rows if r['group'] == group]
        if not items:
            continue
        count = np.asarray([float(r['sample_count']) for r in items], dtype=np.float64)
        k_std = np.asarray([float(r['bootstrap_k_std']) for r in items], dtype=np.float64)
        mode = np.asarray([float(r['bootstrap_mode_fraction']) for r in items], dtype=np.float64)
        sil = np.asarray([float(r['best_silhouette']) for r in items], dtype=np.float64)
        out.append({
            'group': group,
            'num_classes': len(items),
            'sample_count_mean': float(np.nanmean(count)),
            'bootstrap_k_std_mean': float(np.nanmean(k_std)),
            'bootstrap_mode_fraction_mean': float(np.nanmean(mode)),
            'best_silhouette_mean': float(np.nanmean(sil)),
            'interpretation': 'lower k_std and higher mode_fraction means more stable cluster-count estimation',
        })
    return out


def build_correlation_rows(rows):
    counts = np.asarray([float(r['sample_count']) for r in rows], dtype=np.float64)
    out = []
    for metric in ['intra_variance', 'estimated_k', 'best_silhouette', 'bootstrap_k_std', 'bootstrap_mode_fraction']:
        vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        out.append({
            'x': 'sample_count',
            'y': metric,
            'pearson': pearson(counts, vals),
            'spearman': spearman(counts, vals),
        })
    return out


def save_metric_plots(rows, figure_dir, prefix):
    ensure_dir(figure_dir)
    counts = np.asarray([float(r['sample_count']) for r in rows], dtype=np.float64)
    group_colors = {'many': '#1f77b4', 'medium': '#ff7f0e', 'few': '#2ca02c'}
    colors = [group_colors[r['group']] for r in rows]
    for metric in ['intra_variance', 'estimated_k', 'best_silhouette']:
        vals = np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
        mask = np.isfinite(vals)
        plt.figure(figsize=(6, 4))
        plt.scatter(counts[mask], vals[mask], c=np.asarray(colors, dtype=object)[mask], s=24, alpha=0.85)
        plt.xscale('log')
        plt.xlabel('Class sample count (log scale)')
        plt.ylabel(metric)
        plt.title(f'{prefix}: sample support vs {metric}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f'structure_{prefix}_{metric}.png'), dpi=180)
        plt.close()


def save_stability_plot(rows, figure_dir, prefix):
    ensure_dir(figure_dir)
    groups = ['many', 'medium', 'few']
    means = []
    modes = []
    for group in groups:
        items = [r for r in rows if r['group'] == group]
        means.append(float(np.nanmean([float(r['bootstrap_k_std']) for r in items])))
        modes.append(float(np.nanmean([float(r['bootstrap_mode_fraction']) for r in items])))

    x = np.arange(len(groups))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(x - width / 2, means, width, label='bootstrap K std', color='#4c78a8')
    ax1.set_ylabel('Bootstrap K std')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, modes, width, label='mode fraction', color='#f58518')
    ax2.set_ylabel('Mode fraction')
    ax1.set_title(f'{prefix}: cluster-count stability by group')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, f'structure_{prefix}_cluster_stability.png'), dpi=180)
    plt.close(fig)


def select_representative_classes(rows, per_group=2):
    selected = []
    for group in ['many', 'medium', 'few']:
        group_rows = [r for r in rows if r['group'] == group]
        group_rows = sorted(
            group_rows,
            key=lambda r: (
                -float(r['best_silhouette']) if np.isfinite(float(r['best_silhouette'])) else 1.0,
                -int(r['sample_count']),
                int(r['class_id']),
            ),
        )
        selected.extend([int(r['class_id']) for r in group_rows[:per_group]])
    return selected


def save_tsne_plot(features, targets, rows, figure_dir, prefix, seed):
    ensure_dir(figure_dir)
    rng = np.random.default_rng(seed)
    selected = select_representative_classes(rows, per_group=2)
    idx_parts = []
    for class_id in selected:
        idx = np.where(targets == class_id)[0]
        if len(idx) > 180:
            idx = rng.choice(idx, size=180, replace=False)
        idx_parts.append(idx)
    if not idx_parts:
        return
    idx = np.concatenate(idx_parts)
    if len(idx) < 10:
        return
    perplexity = min(30, max(5, (len(idx) - 1) // 3))
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=seed,
    ).fit_transform(features[idx])

    plt.figure(figsize=(7, 5))
    cmap = plt.get_cmap('tab10')
    for color_id, class_id in enumerate(selected):
        class_mask = targets[idx] == class_id
        row = next(r for r in rows if int(r['class_id']) == class_id)
        label = f'c{class_id} {row["group"]} n={row["sample_count"]} K={row["estimated_k"]}'
        plt.scatter(xy[class_mask, 0], xy[class_mask, 1], s=12, alpha=0.75, color=cmap(color_id), label=label)
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.title(f'{prefix}: representative intra-class structures')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'structure_{prefix}_tsne_representative_classes.png'), dpi=180)
    plt.close()


def save_pca_plot(features, targets, rows, figure_dir, prefix, seed):
    ensure_dir(figure_dir)
    rng = np.random.default_rng(seed)
    selected = []
    for group in ['many', 'medium', 'few']:
        group_rows = [r for r in rows if r['group'] == group]
        if not group_rows:
            continue
        group_rows = sorted(group_rows, key=lambda r: (-int(r['sample_count']), int(r['class_id'])))
        selected.append(int(group_rows[0]['class_id']))
    mask = np.isin(targets, selected)
    idx = np.where(mask)[0]
    if len(idx) > 1200:
        idx = rng.choice(idx, size=1200, replace=False)
    xy = PCA(n_components=2, random_state=seed).fit_transform(features[idx])
    plt.figure(figsize=(6, 5))
    for class_id in selected:
        class_mask = targets[idx] == class_id
        row = next(r for r in rows if int(r['class_id']) == class_id)
        label = f'c{class_id} {row["group"]} n={row["sample_count"]}'
        plt.scatter(xy[class_mask, 0], xy[class_mask, 1], s=12, alpha=0.75, label=label)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title(f'{prefix}: representative class structure')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f'structure_{prefix}_pca_classes.png'), dpi=180)
    plt.close()


def main():
    cli = parse_args()
    run_dir = os.path.join(cli.runs_dir, cli.run_name)
    checkpoint_path = os.path.join(run_dir, 'best.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')

    train_args = load_training_args(run_dir)
    if cli.data_root is not None:
        train_args.data_root = cli.data_root
    train_set, _, num_classes, cls_num_list = build_dataset(
        train_args.dataset,
        train_args.data_root,
        imb_factor=train_args.imb_factor,
        seed=train_args.seed,
        synthetic_size=train_args.synthetic_size,
        num_classes=train_args.num_classes,
    )
    set_subset_transform(train_set, deterministic_train_transform(train_args.dataset))

    device_name = cli.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        device_name = 'cpu'
    device = torch.device(device_name)
    model = build_model(train_args, num_classes, cls_num_list).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    loader = DataLoader(
        train_set,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.workers,
        pin_memory=(device.type == 'cuda'),
    )
    features, targets = extract_features(model, loader, device)

    many, medium, few = split_class_groups(cls_num_list)
    rows = []
    for class_id in range(num_classes):
        x = features[targets == class_id]
        estimated_k, best_sil = estimate_cluster_count(
            x, cli.max_clusters, cli.min_cluster_samples, cli.seed + class_id
        )
        boot_std, boot_mode = bootstrap_cluster_stability(
            x,
            cli.max_clusters,
            cli.min_cluster_samples,
            cli.bootstrap_iters,
            cli.bootstrap_size,
            cli.seed + 10_000 + class_id,
        )
        if class_id in many:
            group = 'many'
        elif class_id in medium:
            group = 'medium'
        else:
            group = 'few'
        rows.append({
            'class_id': class_id,
            'sample_count': int(cls_num_list[class_id]),
            'group': group,
            'intra_variance': intra_class_variance(x),
            'estimated_k': int(estimated_k),
            'best_silhouette': best_sil,
            'bootstrap_k_std': boot_std,
            'bootstrap_mode_fraction': boot_mode,
        })

    prefix = cli.output_prefix or cli.run_name
    diag_path = os.path.join(cli.output_dir, f'structure_diagnostics_{prefix}.csv')
    group_path = os.path.join(cli.output_dir, f'structure_group_summary_{prefix}.csv')
    corr_path = os.path.join(cli.output_dir, f'structure_correlations_{prefix}.csv')
    stability_path = os.path.join(cli.output_dir, f'structure_cluster_stability_{prefix}.csv')

    diag_fields = [
        'class_id', 'sample_count', 'group', 'intra_variance',
        'estimated_k', 'best_silhouette', 'bootstrap_k_std',
        'bootstrap_mode_fraction',
    ]
    write_csv(diag_path, rows, diag_fields)
    group_rows = summarize_groups(rows)
    write_csv(group_path, group_rows, list(group_rows[0].keys()))
    corr_rows = build_correlation_rows(rows)
    write_csv(corr_path, corr_rows, list(corr_rows[0].keys()))
    stability_rows = build_stability_rows(rows)
    write_csv(stability_path, stability_rows, list(stability_rows[0].keys()))
    save_metric_plots(rows, cli.figure_dir, prefix)
    save_stability_plot(rows, cli.figure_dir, prefix)
    save_pca_plot(features, targets, rows, cli.figure_dir, prefix, cli.seed)
    save_tsne_plot(features, targets, rows, cli.figure_dir, prefix, cli.seed)

    print(f'Wrote {diag_path}')
    print(f'Wrote {group_path}')
    print(f'Wrote {corr_path}')
    print(f'Wrote {stability_path}')


if __name__ == '__main__':
    main()
