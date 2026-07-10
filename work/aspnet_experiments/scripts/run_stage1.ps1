$ErrorActionPreference = "Stop"

python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model ce --epochs 200 --batch-size 128 --lr 0.1 --device cuda --run-name cifar100lt_if100_ce
python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model proto --proto-mode single --epochs 200 --batch-size 128 --lr 0.1 --device cuda --run-name cifar100lt_if100_single_proto
python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model proto --proto-mode fixed --fixed-k 4 --epochs 200 --batch-size 128 --lr 0.1 --device cuda --run-name cifar100lt_if100_fixed_k4
python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model adaptive_proto --k-max 4 --epochs 200 --batch-size 128 --lr 0.1 --device cuda --run-name cifar100lt_if100_adaptive_k4
