$ErrorActionPreference = "Stop"
python train.py --dataset synthetic --model adaptive_proto --num-classes 10 --epochs 1 --batch-size 64 --synthetic-size 512 --device cpu --run-name smoke_adaptive_proto
