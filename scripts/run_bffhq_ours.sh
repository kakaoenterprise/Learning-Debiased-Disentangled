# BFFHQ
# 0.5pct
python train.py --dataset bffhq --exp=bffhq_0.5_ours --lr=0.0001 --percent=0.5pct --lambda_swap=0.1 --curr_step=10000 --use_lr_decay --lr_decay_step=10000 --lambda_dis_align 2. --lambda_swap_align 2. --dataset bffhq --train_ours --tensorboard --wandb