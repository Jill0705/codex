# ASPNet Long-Tailed Experiments

This is the first experimental scaffold for ASPNet. The first stage intentionally focuses on the core claim:

Adaptive class-specific sub-prototype budgets improve long-tailed recognition over single-prototype and fixed multi-prototype classifiers.

EMA updates, Hungarian matching, and logit adjustment are intentionally kept out of the first training target. They should be added only after the basic adaptive sub-prototype story is validated.

## Install

cd C:\Users\x2472\Documents\Codex\2026-07-04\new-chat-2\work\aspnet_experiments
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt


## Windows GPU PyTorch Install

Install PyTorch separately before installing the light requirements. The generic requirements file intentionally does not install torch, because Windows CUDA wheels should match your local CUDA/Python setup.

Recommended flow:

1. Create and activate a fresh virtual environment.
2. Install PyTorch from the official selector: https://pytorch.org/get-started/locally/
3. Then run: pip install -r requirements.txt

Example for many NVIDIA Windows setups:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

Check it with:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

## Quick Smoke Test

python train.py --dataset synthetic --model adaptive_proto --num-classes 10 --epochs 1 --batch-size 64 --synthetic-size 512 --device cpu

## CIFAR-100-LT First Runs

python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model ce --epochs 200 --batch-size 128 --lr 0.1 --device cuda

python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model proto --proto-mode single --epochs 200 --batch-size 128 --lr 0.1 --device cuda

python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model proto --proto-mode fixed --fixed-k 4 --epochs 200 --batch-size 128 --lr 0.1 --device cuda

python train.py --dataset cifar100lt --data-root data --imb-factor 100 --model adaptive_proto --k-max 4 --epochs 200 --batch-size 128 --lr 0.1 --device cuda

## Key Comparisons

- CE vs single-prototype classifier
- single prototype vs fixed K prototypes
- fixed K prototypes vs adaptive K prototypes
- class accuracy by Many / Medium / Few groups
- learned class-specific prototype counts for adaptive runs
