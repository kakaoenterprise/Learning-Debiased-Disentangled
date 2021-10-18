# Cifar10c
# 0.5pct
python train.py --dataset cifar10c --exp=cifar10c_0.5_vanilla --lr=0.001 --percent=0.5pct --train_vanilla --tensorboard --wandb

# 1pct
python train.py --dataset cifar10c --exp=cifar10c_1_vanilla --lr=0.001 --percent=1pct --train_vanilla --tensorboard --wandb

# 2pct
python train.py --dataset cifar10c --exp=cifar10c_2_vanilla --lr=0.001 --percent=2pct --train_vanilla --tensorboard --wandb

# 5pct
python train.py --dataset cifar10c --exp=cifar10c_5_vanilla --lr=0.001 --percent=5pct --train_vanilla --tensorboard --wandb
