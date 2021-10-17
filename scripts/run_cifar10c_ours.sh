# Cifar10c

for i in {1..5}
  do
    # 0.5pct
    CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_0.5_ours_1_$i --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
    CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_0.5_ours_2_$i --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &

    # 1pct
    CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_1_ours_1_$i --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
    CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_1_ours_2_$i --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &

    # 2pct
    CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_2_ours_1_$i --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
    CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_2_ours_2_$i --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &

    # 5pct
    CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_5_ours_1_$i --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
    CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_5_ours_2_$i --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb
  done

##### Release scripts #####
# 0.5pct
#CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_0.5_ours_$i --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
#
## 1pct
#CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_1_ours_$i --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
#
## 2pct
#CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_2_ours_$i --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb &
#
## 5pct
#CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_5_ours_$i --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard --wandb
