# Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching
<div align="center">
  <img src="assets/teaser.png" width="100%" ></img>
  <br>
  <em>
      (Results on DiT-XL/2 and U-ViT-H/2) 
  </em>
</div>
<br>

> **Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching**   🥯[[Arxiv]](https://arxiv.org/abs/2312.00858)    
> [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Michael Bi Mi](), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore, Huawei Technologies Ltd  


<div align="center">
  <img src="U-ViT/u-vit.gif" width="30%" ></img>
  <br>
  <em>
      (Changes in the router during training) 
  </em>
</div>

## Introduction
We introduce a novel scheme, named **L**earning-to-**C**ache (L2C), that learns to conduct caching in a dynamic manner for diffusion transformers.

Results:

1. A large proportion of layers in the diffusion transformer can be removed even without updating the model parameters. In the case of U-ViT-H/2, for example, we may remove up to 93.68% of the computation in the cache steps (46.84% for all steps), with less than 0.01 drop in FID. 

2. Experimental results show that L2C largely outperforms samplers such as DDIM and DPM-Solver, alongside prior cache-based methods at the same inference speed. 

## Code
We implement Learning-to-Cache on two basic structures: DiT and U-ViT. Check the code below for instruction:

1. DiT: [README](https://github.com/horseee/learning-to-cache/tree/main/DiT#learning-to-cache-for-dit)
2. U-ViT: [README](https://github.com/horseee/learning-to-cache/blob/main/U-ViT/readme.md)

## Citation