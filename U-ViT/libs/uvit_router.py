import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
import numpy as np

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = False #use_checkpoint

    def forward(self, x, skip=None, reuse_att=None, reuse_mlp=None,
                reuse_att_weight=0, reuse_mlp_weight=0):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, skip, reuse_att, reuse_mlp, 
                reuse_att_weight, reuse_mlp_weight
            )
        else:
            return self._forward(
                x, skip, reuse_att, reuse_mlp,
                reuse_att_weight, reuse_mlp_weight
            )

    def _forward(self, x, skip=None, reuse_att=None, reuse_mlp=None, reuse_att_weight=None, reuse_mlp_weight=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        
        att_out = self.attn(self.norm1(x))
        if reuse_att is not None:
            att_out = att_out * (1 - reuse_att_weight) + reuse_att * reuse_att_weight
        x = x + att_out

        mlp_out = self.mlp(self.norm2(x))
        if reuse_mlp is not None:
            mlp_out = mlp_out * (1 - reuse_mlp_weight) + reuse_mlp * reuse_mlp_weight
        x = x + mlp_out
        return x, (att_out, mlp_out)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Router(nn.Module):
    def __init__(self, num_choises):
        super().__init__()
        self.num_choises = num_choises
        self.prob = torch.nn.Parameter(torch.randn(num_choises), requires_grad=True)
        
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x=None): # Any input will be ignored, only for solving the issue of https://github.com/pytorch/pytorch/issues/37814
        return self.activation(self.prob)

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])
        
        self.depth = depth + 1 # depth//2 for in/out, and 1 for mid

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.reset()

    def reset_cache_features(self):
        self.cache_features = [None] * self.depth
        self.activate_cache = False
        self.record_cache = True

    def reset(self):
        self.cur_step_idx = 0
        self.reset_cache_features()

    def add_router(self, num_nfes):
        self.routers = torch.nn.ModuleList([
            Router(2*self.depth) for _ in range(num_nfes)
        ])
    
    def set_activate_cache(self, activate_cache):
        self.activate_cache = activate_cache

    def set_record_cache(self, record_cache):
        self.record_cache = record_cache

    def set_timestep_map(self, timestep_map):
        self.timestep_map = {timestep: i for i, timestep in enumerate(timestep_map)}
        print("Timestep -> Router IDX Map:", self.timestep_map)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        #print("In Model: Get y: ", y, ". Get Timesteps: ", timesteps)
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []        
        cache_features = self.cache_features
        if self.activate_cache :
            router_idx = self.timestep_map[np.round(timesteps[0].item())]
            scores = self.routers[router_idx]()
            router_l1_loss = scores.sum()
        else:
            router_l1_loss = None

        layer_idx = 0
        for blk in self.in_blocks:
            if cache_features[layer_idx] is not None and self.activate_cache:
                reuse_att, reuse_mlp = cache_features[layer_idx]
                reuse_att_weight = 1 - scores[layer_idx*2]
                reuse_mlp_weight = 1 - scores[layer_idx*2+1]
            else:
                reuse_att, reuse_mlp = None, None
                reuse_att_weight, reuse_mlp_weight = 0, 0 

            x, cache_feature = blk(
                x, reuse_att=reuse_att, reuse_mlp=reuse_mlp,
                reuse_att_weight=reuse_att_weight, 
                reuse_mlp_weight=reuse_mlp_weight,
            ) 
            skips.append(x)
            if self.record_cache:
                cache_features[layer_idx] = cache_feature
            layer_idx += 1

        if cache_features[layer_idx] is not None and self.activate_cache:
            reuse_att, reuse_mlp = cache_features[layer_idx]
            reuse_att_weight = 1 - scores[layer_idx*2]
            reuse_mlp_weight = 1 - scores[layer_idx*2+1]
        else:
            reuse_att, reuse_mlp = None, None
            reuse_att_weight, reuse_mlp_weight = 0, 0

        x, cache_feature = self.mid_block(
            x, reuse_att=reuse_att, reuse_mlp=reuse_mlp,
            reuse_att_weight=reuse_att_weight, 
            reuse_mlp_weight=reuse_mlp_weight,
        ) 
        if self.record_cache:
            cache_features[layer_idx] = cache_feature
        layer_idx += 1

        for blk in self.out_blocks:
            if cache_features[layer_idx] is not None and self.activate_cache:
                reuse_att, reuse_mlp = cache_features[layer_idx]
                reuse_att_weight = 1 - scores[layer_idx*2]
                reuse_mlp_weight = 1 - scores[layer_idx*2+1]
            else:
                reuse_att, reuse_mlp = None, None
                reuse_att_weight, reuse_mlp_weight = 0, 0

            x , cache_feature = blk(
                x, skips.pop(), reuse_att=reuse_att, reuse_mlp=reuse_mlp,
                reuse_att_weight=reuse_att_weight, 
                reuse_mlp_weight=reuse_mlp_weight,
            )
            if self.record_cache:
                cache_features[layer_idx] = cache_feature
            layer_idx += 1

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)

        self.cur_step_idx += 1

        if self.activate_cache:
            return x, router_l1_loss
        else:
            return x
