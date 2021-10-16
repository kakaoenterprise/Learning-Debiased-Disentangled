##  (NeurIPS 2021, Oral): Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> Learning Debiased Representation via Disentangled Feature Augmentation <br>
> [Jungsoo Lee](https://leebebeto.github.io/)* (KAIST AI, Kakao Enterprise), [Eungyeup Kim](https://eungyeupkim.github.io/)* (KAIST AI, Kakao Enterprise), Juyoung Lee (Kakao Enterprise)<br>
> Jihyeon Lee (KAIST AI), and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/) (KAIST AI) (*: equal contribution)<br>
> NeurIPS 2021, Oral<br>

> Paper: [arxiv](https://arxiv.org/abs/2103.15597)<br>

> Youtube Video (English): [Youtube / 5min](https://youtu.be/3vf7Oh6gYEE) <br>

> **Abstract:** 
*Despite the unprecedented improvement of face recognition, existing face recognition models still show considerably low performances in determining whether a pair of child and adult images belong to the same identity.
Previous approaches mainly focused on increasing the similarity between child and adult images of a given identity to overcome the discrepancy of facial appearances due to aging.
However, we observe that reducing the similarity between child images of different identities is crucial for learning distinct features among children and thus improving face recognition performance in child-adult pairs.
Based on this intuition, we propose a novel loss function called the Inter-Prototype loss which minimizes the similarity between child images. 
Unlike the previous studies, the Inter-Prototype loss does not require additional child images or training additional learnable parameters.
Our extensive experiments and in-depth analyses show that our approach outperforms existing baselines in face recognition with child-adult pairs.*<br>

<p align="center">
  <img src="assets/main.png" />
</p>

## Code Contributors
[Jungsoo Lee](https://leebebeto.github.io/) (KAIST AI), [Jooyeol Yun](https://www.linkedin.com/in/jooyeol-yun-6a176a1b6/) (KAIST AI)

## Concept Video
Coming soon!
[comment]: <> (Click the figure to watch the youtube video of our paper!)

[comment]: <> (<p align="center">)

[comment]: <> (  <a href="https://youtu.be/3vf7Oh6gYEE"><img src="assets/robustnet_motivation.png" alt="Youtube Video"></a><br>)

[comment]: <> (</p>)

## Pytorch Implementation
### Installation
Clone this repository.
```
git clone https://github.com/leebebeto/Inter-Prototype.git
cd Inter-Prototype
pip install -r requirements
CUDA_VISIBLE_DEVICES=0 python3 train.py --wandb --tensorboard
```

### How to Run 
We used two different training datasets: 1) [CASIA WebFace]() and 2) [MS1M]().
For the test sets, we constructed test sets with child-adult pairs with at least 20 years and 30 years age gaps using AgeDB and FG-NET, termed as AgeDB-C20, AgeDB-C30, FGNET-C20, and FGNET-C30.
We also used LAG (Large Age Gap) dataset for the test set. 
```
Train
 └ CASIA-WebFace
   └ id1
     └ image1.jpg
     └ image2.jpg
     └ ...
   └ id2
     └ image1.jpg
     └ image2.jpg
     └ ...     
   ...
 └ MS1M
   └ id1
     └ image1.jpg
     └ image2.jpg
     └ ...
   └ id2
     └ image1.jpg
     └ image2.jpg
     └ ...     
   ...
```
```
Test
 └ AgeDB-aligned
   └ image1.jpg
   └ image2.jpg
   └ ...
 └ FGNET-aligned
   └ image1.jpg
   └ image2.jpg
   └ ...
```

### Pretrained Models
#### All models trained for our paper
You can validate pretrained model with following commands.
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --wandb --tensorboard
```

## Quantitative / Qualitative Evaluation
### Trained with CASIA WebFace dataset
<p align="center">
  <img src="assets/casia.png" />
</p>

### Trained with MS1M dataset
<p align="center">
  <img src="assets/ms1m.png" />
</p>

### t-SNE embedding of prototype vectors
<p align="center">
  <img src="assets/tsne.png" />
</p>


### Acknowledgments
Our pytorch implementation is heavily derived from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch).
Thanks for the implementation.
