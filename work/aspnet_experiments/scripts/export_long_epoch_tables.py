import argparse
import ast
import csv
from pathlib import Path


RUNS = [
    {
        'method': 'CE',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': 'N/A',
        'role': 'baseline',
        'runs': {100: 'ce_e100', 200: 'ce_e200'},
    },
    {
        'method': 'Balanced Softmax',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': 'N/A',
        'role': 'strong_baseline',
        'runs': {100: 'balanced_softmax_if100_e100', 200: 'balanced_softmax_if100_e200'},
    },
    {
        'method': 'LDAM-DRW',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': 'N/A',
        'role': 'strong_baseline',
        'runs': {100: 'ldam_drw_if100_e100_s1', 200: 'ldam_drw_if100_e200_s1'},
    },
    {
        'method': 'Fixed K4',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': '400',
        'role': 'prototype_baseline',
        'runs': {100: 'fixed_k4_t01_tau025_e100', 200: 'fixed_k4_t01_tau025_e200'},
    },
    {
        'method': 'Sqrt B200',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': '200',
        'role': 'main_method',
        'runs': {100: 'adaptive_sqrt_budget200_e100', 200: 'adaptive_sqrt_budget200_e200'},
    },
    {
        'method': 'Sqrt B200 + LDAM',
        'dataset': 'CIFAR-100-LT',
        'IF': '100',
        'prototypes': '200',
        'role': 'compatibility',
        'runs': {
            100: 'adaptive_sqrt_budget200_if100_e100_ldam_s1',
            200: 'adaptive_sqrt_budget200_if100_e200_ldam_s1',
        },
    },
    {
        'method': 'CE',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': 'N/A',
        'role': 'baseline',
        'runs': {100: 'ce_if50_e100', 200: 'ce_if50_e200'},
    },
    {
        'method': 'Balanced Softmax',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': 'N/A',
        'role': 'strong_baseline',
        'runs': {100: 'balanced_softmax_if50_e100', 200: 'balanced_softmax_if50_e200'},
    },
    {
        'method': 'LDAM-DRW',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': 'N/A',
        'role': 'strong_baseline',
        'runs': {100: 'ldam_drw_if50_e100_s1', 200: 'ldam_drw_if50_e200_s1'},
    },
    {
        'method': 'Fixed K3',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': '300',
        'role': 'prototype_baseline',
        'runs': {100: 'fixed_k3_if50_e100', 200: 'fixed_k3_if50_e200'},
    },
    {
        'method': 'Fixed K4',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': '400',
        'role': 'prototype_baseline',
        'runs': {100: 'fixed_k4_if50_e100', 200: 'fixed_k4_if50_e200'},
    },
    {
        'method': 'Sqrt B300',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': '300',
        'role': 'main_method',
        'runs': {100: 'adaptive_sqrt_budget300_if50_e100', 200: 'adaptive_sqrt_budget300_if50_e200'},
    },
    {
        'method': 'Sqrt B300 + LDAM',
        'dataset': 'CIFAR-100-LT',
        'IF': '50',
        'prototypes': '300',
        'role': 'compatibility',
        'runs': {
            100: 'adaptive_sqrt_budget300_if50_e100_ldam_s1',
            200: 'adaptive_sqrt_budget300_if50_e200_ldam_s1',
        },
    },
]


FIELDS = [
    'method',
    'dataset',
    'IF',
    'epoch_budget',
    'run_name',
    'status',
    'best_epoch',
    'val',
    'many',
    'medium',
    'few',
    'prototypes',
    'role',
]


def parse_best_metrics(log_path):
    if not log_path.exists():
        return None
    best = None
    with log_path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("{'epoch':"):
                continue
            try:
                row = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if best is None or float(row['val_acc']) > float(best['val_acc']):
                best = row
    return best


def make_row(spec, epoch_budget, logs_dir):
    run_name = spec['runs'][epoch_budget]
    best = parse_best_metrics(logs_dir / f'{run_name}.txt')
    row = {
        'method': spec['method'],
        'dataset': spec['dataset'],
        'IF': spec['IF'],
        'epoch_budget': epoch_budget,
        'run_name': run_name,
        'status': 'missing' if best is None else 'ok',
        'best_epoch': '',
        'val': '',
        'many': '',
        'medium': '',
        'few': '',
        'prototypes': spec['prototypes'],
        'role': spec['role'],
    }
    if best is not None:
        row.update({
            'best_epoch': best['epoch'],
            'val': f"{float(best['val_acc']):.2f}",
            'many': f"{float(best['many_acc']):.2f}",
            'medium': f"{float(best['medium_acc']):.2f}",
            'few': f"{float(best['few_acc']):.2f}",
        })
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path} ({len(rows)} rows)')


def main():
    parser = argparse.ArgumentParser(description='Export paper tables from long-epoch text logs.')
    parser.add_argument('--logs-dir', default='logs')
    parser.add_argument('--tables-dir', default='tables')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    tables_dir = Path(args.tables_dir)
    rows100 = [make_row(spec, 100, logs_dir) for spec in RUNS]
    rows200 = [make_row(spec, 200, logs_dir) for spec in RUNS]
    write_csv(tables_dir / 'ccfb_long_epoch_e100.csv', rows100)
    write_csv(tables_dir / 'ccfb_long_epoch_e200.csv', rows200)
    write_csv(tables_dir / 'ccfb_long_epoch_comparison.csv', rows100 + rows200)


if __name__ == '__main__':
    main()
