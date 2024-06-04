# Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching
<div align="center">
  <img src="assets/teaser.png" width="100%" ></img>
  <br>
  <em>
      (Results on DiT-XL/2 and U-ViT-H/2) 
  </em>
</div>
<br>

> **Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching**   🥯[[Arxiv]]()    
> [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Michael Bi Mi](), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore, Huawei Technologies Ltd  


<div align="center">
  <img src="U-ViT/u-vit.gif" width="40%" ></img>
  <br>
  <em>
      (Changes in the router for U-ViT during training) 
  </em>
</div>

## Introduction
We introduce a novel scheme, named **L**earning-to-**C**ache (L2C), that learns to conduct caching in a dynamic manner for diffusion transformers.

Some takeaways:

1. A large proportion of layers in the diffusion transformer can be removed, without updating the model parameters.
   - In U-ViT-H/2, up to 93.68% of the layers in the cache steps (46.84% for all steps) can be removed, with less than 0.01 drop in FID. 

2. L2C largely outperforms samplers such as DDIM and DPM-Solver. 

## Checkpoint for Routers
| Model | NFE | Checkpoint |
| -- | -- | -- |
| DiT-XL/2 |  50 | [link](DiT/ckpt/DDIM50_router.pt) |
| DiT-XL/2 |  20 | [link](DiT/ckpt/DDIM20_router.pt) |
| U-ViT-H/2 |  50 | [link](U-ViT/ckpt/dpm50_router.pth) |
| U-ViT-H/2 |  20 | [link](U-ViT/ckpt/dpm20_router.pth)|

## Code
We implement Learning-to-Cache on two basic structures: DiT and U-ViT. Check the instructions below:

1. DiT: [README](https://github.com/horseee/learning-to-cache/tree/main/DiT#learning-to-cache-for-dit)
2. U-ViT: [README](https://github.com/horseee/learning-to-cache/blob/main/U-ViT/readme.md)

## Citation
```
TBD
```
