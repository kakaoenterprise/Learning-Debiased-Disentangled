##  (NeurIPS 2021, Oral): Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> Learning Debiased Representation via Disentangled Feature Augmentation <br>
> [Jungsoo Lee](https://leebebeto.github.io/)* (KAIST AI, Kakao Enterprise), [Eungyeup Kim](https://eungyeupkim.github.io/)* (KAIST AI, Kakao Enterprise), Juyoung Lee (Kakao Enterprise)<br>
> Jihyeon Lee (KAIST AI), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/) (KAIST AI) (*: equal contribution)<br>
> NeurIPS 2021, Oral<br>

> Paper: [arxiv](https://arxiv.org/abs/2103.15597)<br>

> **Abstract:** 
*Image classification models tend to make decisions based on peripheral attributes of data items that have strong correlation with a target variable (i.e., dataset bias).
These biased models suffer from the poor generalization capability when evaluated on unbiased datasets.
Existing approaches for debiasing often identify and emphasize those samples with no such correlation (i.e., bias-conflicting) without defining the bias type in advance.
However, such bias-conflicting samples are significantly scarce in biased datasets, limiting the debiasing capability of these approaches.
This paper first presents an empirical analysis revealing that training with "diverse" bias-conflicting samples beyond a given training set is crucial for debiasing as well as the generalization capability.
Based on this observation, we propose a novel feature-level data augmentation technique in order to synthesize diverse bias-conflicting samples. 
To this end, our method learns the disentangled representation of (1) the intrinsic attributes (i.e., those inherently defining a certain class) and (2) bias attributes (i.e., peripheral attributes causing the bias), from a large number of bias-aligned samples, the bias attributes of which have strong correlation with the target variable. 
Using the disentangled representation, we synthesize bias-conflicting samples that contain the diverse intrinsic attributes of bias-aligned samples by swapping their latent features.
By utilizing these diversified bias-conflicting features during the training, our approach achieves superior classification accuracy and debiasing results against the existing baselines on both synthetic as well as a real-world dataset.*<br>

<p align="center">
  <img src="assets/main.png" />
</p>

## Code Contributors
[Jungsoo Lee](https://leebebeto.github.io/) (KAIST AI, Kakao Enterprise), [Eungyeup Kim](https://eungyeupkim.github.io/) (KAIST AI, Kakao Enterprise), Juyoung Lee (Kakao Enterprise)

## Concept Video
Click the figure to watch the youtube video of our paper!

<p align="center">

  <a href="https://youtu.be/3vf7Oh6gYEE"><img src="assets/robustnet_motivation.png" alt="Youtube Video"></a><br>

</p>

## Pytorch Implementation
### Installation
Clone this repository.
```
git clone https://github.com/kakaoenterprise/NeurIPS2021/Learning-Debiased-Disentangled.git
cd Learning-Debiased-Disentangled
pip install -r requirements
```
### Datasets
We used three datasets in our paper. 

<p align="center">
  <img src="assets/data.png" />
</p>


### How to Run
#### CMNIST
##### Vanilla
```
python train.py --dataset cmnist --exp=cmnist_0.5_vanilla --lr=0.01 --percent=0.5pct --train_vanilla --tensorboard
python train.py --dataset cmnist --exp=cmnist_1_vanilla --lr=0.01 --percent=1pct --train_vanilla --tensorboard
python train.py --dataset cmnist --exp=cmnist_2_vanilla --lr=0.01 --percent=2pct --train_vanilla --tensorboard
python train.py --dataset cmnist --exp=cmnist_5_vanilla --lr=0.01 --percent=5pct --train_vanilla --tensorboard
```
##### Ours
```
python train.py --dataset cmnist --exp=cmnist_0.5_ours --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=10 --lambda_align_swap=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cmnist --exp=cmnist_1_ours --lr=0.01 --percent=1pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=10 --lambda_align_swap=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cmnist --exp=cmnist_2_ours --lr=0.01 --percent=2pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=10 --lambda_align_swap=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cmnist --exp=cmnist_5_ours --lr=0.01 --percent=5pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=10 --lambda_align_swap=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
```

#### Corrupted CIFAR10
##### Vanilla
```
python train.py --dataset cifar10c --exp=cifar10c_0.5_vanilla --lr=0.001 --percent=0.5pct --train_vanilla --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_1_vanilla --lr=0.001 --percent=1pct --train_vanilla --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_2_vanilla --lr=0.001 --percent=2pct --train_vanilla --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_5_vanilla --lr=0.001 --percent=5pct --train_vanilla --tensorboard
```
##### Ours
```
python train.py --dataset cifar10c --exp=cifar10c_0.5_ours --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=1 --lambda_align_swap=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_1_ours --lr=0.001 --percent=1pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=5 --lambda_align_swap=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_2_ours --lr=0.001 --percent=2pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=5 --lambda_align_swap=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
python train.py --dataset cifar10c --exp=cifar10c_5_ours --lr=0.001 --percent=5pct --curr_step=10000 --lambda_augment=1 --lambda_align_main=1 --lambda_align_swap=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_ours --tensorboard
```

#### BFFHQ
##### Vanilla
```
python train.py --dataset bffhq --exp=bffhq_0.5_vanilla --lr=0.0001 --percent=0.5pct --train_vanilla --tensorboard
```

##### Ours
```
python train.py --dataset bffhq --exp=bffhq_0.5_ours --lr=0.0001 --percent=0.5pct --lambda_main=1 --lambda_augment=0.1 --curr_step=10000 --use_lr_decay --lr_decay_step=10000 --lambda_align_main 2. --lambda_align_swap 2. --dataset bffhq --train_ours --tensorboard
```


### Acknowledgments
Our pytorch implementation is heavily derived from [LfF](https://github.com/alinlab/LfF).
Thanks for the implementation.
