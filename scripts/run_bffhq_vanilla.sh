# BFFHQ
for i in {1..3}
  do
    python train.py --dataset bffhq --exp=bffhq_0.5_vanilla --lr=0.0001 --percent=0.5pct --train_vanilla --tensorboard --wandb &
  done