
## Preparation

Please follow [U-ViT](https://github.com/baofff/U-ViT) to:
1. Prepara the environment and install necessary packages
2. Download the autoencoder and the reference statistic for FID in `assets/`
3. Download the model [imagenet 256x256(U-ViT-H/2)](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=share_link) and put it here.

After completing the above steps, those files would be contained in the directory:
```
- imagenet256_uvit_huge.pth
- assets 
  | - fid_stats
      | - fid_stats_imagenet256_guided_diffusion.npz
      | - ...
  | - stable-diffusion
      | - autoencoder_kl_ema.pth
      | - autoencoder_kl.pth
```

## Sample Images
For 20 NFEs in DPM-Solver:
```bash
python sample_ldm_discrete.py --config configs/imagenet256_uvit_huge_dynamic_cache.py --nnet_path imagenet256_uvit_huge.pth --nfe 20 --router ckpt/dpm20_router.pth --thres 0.9
```

For 50 NFEs in DPM-Solver:
```bash
python sample_ldm_discrete.py --config configs/imagenet256_uvit_huge_dynamic_cache.py --nnet_path imagenet256_uvit_huge.pth --nfe 50 --router ckpt/dpm50_router.pth --thres 0.9
```

The code would repeat the generation for 5 times to avoid the fluctuations in the inference time. If you want to see the images without acceleration, you can use the follwing command:

```bash
python sample_ldm_discrete.py --config configs/imagenet256_uvit_huge.py --nnet_path imagenet256_uvit_huge.pth --nfe 50
```

## Sample 50k Images for Evaluation

```bash
export NFE=50
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet256_uvit_huge_dynamic_cache.py --nnet_path=imagenet256_uvit_huge.pth --config.sample.path=samples/dpm${NFE}_router --nfe=$NFE --router ckpt/dpm${NFE}_router.pth --thres 0.9
```

The FID would be automatically evaluated after the images are all sampled. Be sure to modify NUM_STEPS and PATH_TO_TRAINED_ROUTER to correspond to the respective NFE steps and the location of the router.

Results:

| NFE | Router | FID | 
| -- | -- | -- | 
| 50 | - | 2.3728 | 
| 50 | ckpt/dpm50_router.pth | 2.3625 |
| 20 | - | 2.5739 | 
| 20 | ckpt/dpm20_router.pth |  2.5809|


## Train the router
Execute the following command to train the router:
```
accelerate launch --multi_gpu --main_process_port 18100 --num_processes 8 --mixed_precision fp16 train_router_discrete.py --config=configs/imagenet256_uvit_huge_router.py --config.dataset.path=PATH_TO_IMAGENET   --nnet_path=imagenet256_uvit_huge.pth --nfe=20 --router_lr=0.001 --l1_weight=0.1 --workdir=workdir/uvit_router_l1_0.1
```
Change `PATH_TO_IMAGENET` to your path to the imagenet dataset.

<div align="center">
  <img src="u-vit.gif" width="70%" ></img>
  <br>
  <em>
      (Changes in the router during training) 
  </em>
</div>


## Acknowledgement
This implementation is based on [U-ViT](https://github.com/baofff/U-ViT)