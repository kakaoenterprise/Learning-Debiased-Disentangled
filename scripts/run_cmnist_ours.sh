# CMNIST
# 0.5pct
python train.py --dataset cmnist --exp=cmnist_0.5_ours --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb

# 1pct
python train.py --dataset cmnist --exp=cmnist_1_ours --lr=0.01 --percent=1pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb

# 2pct
python train.py --dataset cmnist --exp=cmnist_2_ours --lr=0.01 --percent=2pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb

# 5pct
python train.py --dataset cmnist --exp=cmnist_5_ours --lr=0.01 --percent=5pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb