# CMNIST
# 0.5pct
for i in {1..3}
  do
    python train.py --dataset cmnist --exp=cmnist_0.5_vanilla --lr=0.01 --percent=0.5pct --train_vanilla --tensorboard --wandb &
  done

# 1pct
for i in {1..3}
  do
    python train.py --dataset cmnist --exp=cmnist_1_vanilla --lr=0.01 --percent=1pct --train_vanilla --tensorboard --wandb &
  done

# 2pct
for i in {1..3}
  do
    python train.py --dataset cmnist --exp=cmnist_2_vanilla --lr=0.01 --percent=2pct --train_vanilla --tensorboard --wandb &
  done

# 5pct
for i in {1..3}
  do
    python train.py --dataset cmnist --exp=cmnist_5_vanilla --lr=0.01 --percent=5pct --train_vanilla --tensorboard --wandb &
  done
