

## Requirement
With pytorch(>2.0) installed, execute the following command to install necessary packages
```
pip install accelerate diffusers timm torchvision wandb
```

## Train the router
```
torchrun --nnodes=1 --nproc_per_node=4 --master_port 12345 train_router.py --model DiT-XL/2 --data-path PATH_TO_IMAGENET_TRAIN --global-batch-size 64 --image-size 256 --ckpt-every 1000 --l1 5e-6 --lr 0.01 --wandb
```
The code would automatically download the pretrained DiT-XL model. Ideally the trained router would be saved in the path: `results/XXX-DiT-XL-2/0020000.pt`


## Sample Image
```
python sample.py --model DiT-XL/2 --num-sampling-steps 50 --ddim-sample --accelerate-method dynamiclayer --path ckpt/DDIM50_router.pt --thres 0.1
```

## Evaluate FID on ImageNet
Put the above path into the following command to sample images.
```
torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 sample_ddp.py --model DiT-XL/2 --num-sampling-steps 20 --ddim-sample --accelerate-method dynamiclayer --path PATH_TO_TRAINED_ROUTER --thres 0.1
```

## Calculate FID
We follow DiT to evaluate FID by [the code](https://github.com/openai/guided-diffusion/tree/main/evaluations). Please install tensorflow, scipy, requests and tqdm first, and then run the following command:
```
python evaluator.py ~/ckpt/VIRTUAL_imagenet256_labeled.npz PATH_TO_SAMPLE_NPZ
```