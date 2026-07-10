import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from aspnet_lt.classifiers import LinearHead, RecognitionModel, SubPrototypeHead
from aspnet_lt.data import build_dataset
from aspnet_lt.losses import build_loss
from aspnet_lt.resnet_cifar import resnet32
from aspnet_lt.utils import AverageMeter, accuracy, append_csv, classwise_accuracy, ensure_dir, group_accuracy, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description='ASPNet stage-1 long-tailed experiments')
    p.add_argument('--dataset', default='cifar100lt', choices=['synthetic', 'cifar10lt', 'cifar100lt'])
    p.add_argument('--data-root', default='data')
    p.add_argument('--imb-factor', type=float, default=100)
    p.add_argument('--model', default='adaptive_proto', choices=['ce', 'proto', 'adaptive_proto'])
    p.add_argument('--proto-mode', default='single', choices=['single', 'fixed'])
    p.add_argument('--fixed-k', type=int, default=4)
    p.add_argument('--allocation', default='log', choices=['log', 'sqrt', 'linear', 'effective'])
    p.add_argument('--proto-budget', type=int, default=None)
    p.add_argument('--k-min', type=int, default=1)
    p.add_argument('--k-max', type=int, default=4)
    p.add_argument('--effective-beta', type=float, default=0.9999)
    p.add_argument('--pooling', default='lse', choices=['lse', 'max', 'mean'])
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--pool-tau', type=float, default=0.25)
    p.add_argument('--prototype-update', default='grad', choices=['grad', 'ema', 'conf_ema'])
    p.add_argument('--ema-momentum', type=float, default=0.9)
    p.add_argument('--confidence-temp', type=float, default=4.0)
    p.add_argument('--logit-adjustment', action='store_true')
    p.add_argument('--la-tau', type=float, default=1.0)
    p.add_argument('--loss', default='ce', choices=['ce', 'balanced_softmax', 'ldam'])
    p.add_argument('--drw-epoch', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--feature-dim', type=int, default=64)
    p.add_argument('--synthetic-size', type=int, default=1024)
    p.add_argument('--num-classes', type=int, default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--runs-dir', default='runs')
    return p.parse_args()


def build_model(args, num_classes, cls_num_list):
    backbone = resnet32(feature_dim=args.feature_dim)
    if args.model == 'ce':
        head = LinearHead(args.feature_dim, num_classes)
    elif args.model == 'proto':
        if args.proto_mode == 'single':
            head = SubPrototypeHead.single(args.feature_dim, num_classes, temperature=args.temperature, pooling=args.pooling, pool_tau=args.pool_tau)
        else:
            head = SubPrototypeHead.fixed(args.feature_dim, num_classes, fixed_k=args.fixed_k, temperature=args.temperature, pooling=args.pooling, pool_tau=args.pool_tau)
    elif args.model == 'adaptive_proto':
        head = SubPrototypeHead.adaptive(
            args.feature_dim,
            num_classes,
            cls_num_list,
            k_max=args.k_max,
            k_min=args.k_min,
            allocation=args.allocation,
            proto_budget=args.proto_budget,
            effective_beta=args.effective_beta,
            temperature=args.temperature,
            pooling=args.pooling,
            pool_tau=args.pool_tau,
        )
    else:
        raise ValueError(args.model)
    if hasattr(head, 'configure_memory_update'):
        head.configure_memory_update(
            mode=args.prototype_update,
            ema_momentum=args.ema_momentum,
            confidence_temp=args.confidence_temp,
        )
    return RecognitionModel(backbone, head)


def make_log_prior(cls_num_list, device):
    counts = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
    prior = counts / counts.sum().clamp_min(1.0)
    return torch.log(prior.clamp_min(1e-12))


def maybe_adjust_logits(logits, log_prior, tau):
    if log_prior is None:
        return logits
    return logits - tau * log_prior.view(1, -1)


def run_epoch(model, loader, criterion, optimizer, device, train=True, epoch=0, log_prior=None, la_tau=1.0):
    model.train(train)
    if train and hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    losses = AverageMeter()
    accs = AverageMeter()
    desc = f'epoch {epoch:03d} ' + ('train' if train else 'eval')
    iterator = tqdm(loader, desc=desc, leave=False)
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        features = model.backbone(images)
        logits = model.head(features)
        metric_logits = maybe_adjust_logits(logits, log_prior, la_tau)
        loss_logits = metric_logits if log_prior is not None else logits
        loss = criterion(loss_logits, targets)
        if train:
            loss.backward()
            optimizer.step()
            if hasattr(model.head, 'update_memory'):
                model.head.update_memory(features, targets)
        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        accs.update(accuracy(metric_logits.detach(), targets), batch_size)
        iterator.set_postfix(loss=f'{losses.avg:.4f}', acc=f'{accs.avg:.2f}')
    return losses.avg, accs.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, cls_num_list, log_prior=None, la_tau=1.0):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    all_logits = []
    all_targets = []
    for images, targets in tqdm(loader, desc='eval', leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        metric_logits = maybe_adjust_logits(logits, log_prior, la_tau)
        loss_logits = metric_logits if log_prior is not None else logits
        loss = criterion(loss_logits, targets)
        losses.update(loss.item(), targets.size(0))
        accs.update(accuracy(metric_logits, targets), targets.size(0))
        all_logits.append(metric_logits.cpu())
        all_targets.append(targets.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    class_acc, _ = classwise_accuracy(logits, targets, num_classes)
    groups = group_accuracy(class_acc, cls_num_list)
    return losses.avg, accs.avg, groups


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    train_set, val_set, num_classes, cls_num_list = build_dataset(
        args.dataset,
        args.data_root,
        imb_factor=args.imb_factor,
        seed=args.seed,
        synthetic_size=args.synthetic_size,
        num_classes=args.num_classes,
    )
    args.num_classes = num_classes
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f'{args.dataset}_{args.model}_{args.proto_mode}_if{int(args.imb_factor)}_seed{args.seed}_{stamp}'
    run_dir = os.path.join(args.runs_dir, run_name)
    ensure_dir(run_dir)
    save_json(vars(args), os.path.join(run_dir, 'args.json'))
    save_json({'cls_num_list': cls_num_list}, os.path.join(run_dir, 'class_counts.json'))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    model = build_model(args, num_classes, cls_num_list).to(device)
    if hasattr(model.head, 'proto_counts'):
        save_json({'proto_counts': model.head.proto_counts, 'total_prototypes': int(sum(model.head.proto_counts))}, os.path.join(run_dir, 'prototype_counts.json'))

    criterion = build_loss(args.loss, cls_num_list, drw_epoch=args.drw_epoch).to(device)
    log_prior = make_log_prior(cls_num_list, device) if args.logit_adjustment else None
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, epoch=epoch, log_prior=log_prior, la_tau=args.la_tau
        )
        val_loss, val_acc, groups = evaluate(
            model, val_loader, criterion, device, num_classes, cls_num_list,
            log_prior=log_prior, la_tau=args.la_tau
        )
        scheduler.step()
        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            **groups,
        }
        append_csv(os.path.join(run_dir, 'metrics.csv'), row)
        print(row)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'args': vars(args), 'best_acc': best_acc}, os.path.join(run_dir, 'best.pt'))
    print(f'Best val acc: {best_acc:.2f}')


if __name__ == '__main__':
    main()
