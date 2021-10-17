# CMNIST
# 0.5pct
python train.py --dataset cmnist --exp=cmnist_0.5_vanilla --lr=0.01 --percent=0.5pct --train_vanilla --tensorboard --wandb

# 1pct
python train.py --dataset cmnist --exp=cmnist_1_vanilla --lr=0.01 --percent=1pct --train_vanilla --tensorboard --wandb

# 2pct
python train.py --dataset cmnist --exp=cmnist_2_vanilla --lr=0.01 --percent=2pct --train_vanilla --tensorboard --wandb

# 5pct
python train.py --dataset cmnist --exp=cmnist_5_vanilla --lr=0.01 --percent=5pct --train_vanilla --tensorboard --wandb
