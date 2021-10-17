# BFFHQ
# 0.5pct
for i in {1..5}
  do
    CUDA_VISIBLE_DEVICES=0 python train.py --dataset bffhq --exp=bffhq_0.5_ours_3_$i --lr=0.0001 --percent=0.5pct --lambda_swap=0.1 --curr_step=10000 --use_lr_decay --lr_decay_step=10000 --lambda_dis_align 2. --lambda_swap_align 2. --dataset bffhq --train_ours --tensorboard --wandb &
    CUDA_VISIBLE_DEVICES=1 python train.py --dataset bffhq --exp=bffhq_0.5_ours_4_$i --lr=0.0001 --percent=0.5pct --lambda_swap=0.1 --curr_step=10000 --use_lr_decay --lr_decay_step=10000 --lambda_dis_align 2. --lambda_swap_align 2. --dataset bffhq --train_ours --tensorboard --wandb
  done