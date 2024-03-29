import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Code taken and modified from facebooks TimeSformer 
# https://github.com/facebookresearch/TimeSforme
from itertools import repeat
from functools import partial
import collections
import math
import warnings
from einops import rearrange, reduce, repeat
from torch import einsum
import itertools


# Utils
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# From PyTorch internals
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(itertools.repeat(x, n))

    parse.__name__ = name
    return parse
to_2tuple = _ntuple(2)

# Code taken and modified from facebooks TimeSformer 
# https://github.com/facebookresearch/TimeSformer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x


class GameFrameEmbed(nn.Module):
  # Image + controlInput to Patch Embedding
  def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768, control_dim=6):
    super().__init__()
    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) + 1
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.cProj = nn.Linear(control_dim, embed_dim)

  def forward(self, x, cont):
    B, C, T, H, W = x.shape
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = self.proj(x)
    W = x.size(-1)
    x = x.flatten(2).transpose(1,2)

    # Create embeddings for controls
    cont = rearrange(cont, 'b c w -> (b c) w')
    cont = self.cProj(cont)
    cont = cont.unsqueeze(1)

    # Stack the control embeddings on top
    x = torch.cat((x,cont), dim=1)

    # (B T) W e
    # B T W E
    x =  rearrange(x, '(b t) w e -> b t w e', b=B, t=T)
    W = x.size(-2)
    return x, T, W 

class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    mlp_hidden_dim = int(dim * mlp_ratio)
    # Space Attention
    self.norm1 = norm_layer(dim)
    self.spaceAttention = Attention(
      dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    self.norm2 = norm_layer(dim)
    self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # Time Attention
    self.norm3 = norm_layer(dim)
    self.timeAttention = Attention(
        dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    self.norm4 = norm_layer(dim)
    self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

  def forward(self, x, B, T, W):
    # Begin with 'B T W E'
    # Space Attention
    x = rearrange(x, 'b t w e -> (b t) w e', b=B,t=T,w=W)
    x = x + self.drop_path(self.spaceAttention(self.norm1(x)))
    x = x + self.drop_path(self.mlp1(self.norm2(x)))

    # Time Attention
    x = rearrange(x, '(b t) w e -> (b w) t e', b=B,t=T,w=W)
    x = x + self.drop_path(self.timeAttention(self.norm3(x)))
    x = x + self.drop_path(self.mlp2(self.norm4(x)))

    # reshape back
    x = rearrange(x, '(b w) t e -> b t w e', b=B,t=T,w=W)
    return x

class SimpleDecoder(nn.Module):
  def __init__(self, emb_size, hidden_size, output_patch_dim):
      super().__init__()
      self.output_patch_dim = output_patch_dim
      self.fc1 = nn.Linear(emb_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, 3*output_patch_dim*output_patch_dim)
      
      self.relu = nn.ReLU()
      
  def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.relu(x)
      x = self.fc3(x)
      x = torch.tanh(x)
      return x.view(x.size(0), 3, self.output_patch_dim, self.output_patch_dim)

class GameEmulator(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, control_size=6, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, dropout=0.):
      super().__init__()
      #self.attention_type = attention_type
      self.img_size = img_size
      self.depth = depth
      self.dropout = nn.Dropout(dropout)

      self.embed_dim = embed_dim 
      self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
      self.frame_embed = GameFrameEmbed( img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, control_dim=control_size)
      num_patches = self.frame_embed.num_patches

      ## Positional Embeddings
      # IDK here!
      self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
      self.pos_drop = nn.Dropout(p=drop_rate)
      self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
      self.time_drop = nn.Dropout(p=drop_rate)

      ## Attention Blocks
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
      self.blocks = nn.ModuleList([
          Block(
              dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
          for i in range(self.depth)])

      self.norm = norm_layer(embed_dim)
      # Image Decoder
      self.decoder = SimpleDecoder(embed_dim, embed_dim*2, patch_size)

      # Weight Initialization
      trunc_normal_(self.pos_embed, std=.02)
      trunc_normal_(self.time_embed, std=.02)
      self.apply(self._init_weights)

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
      return {'pos_embed', 'cls_token', 'time_embed'}

  def get_classifier(self):
      return self.head

  def reset_classifier(self, num_classes, global_pool=''):
      self.num_classes = num_classes
      self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

  def forward_features(self, x, cont):
      B = x.shape[0]
      x, T, W = self.frame_embed(x, cont)

      # How to do this?? How does the encoding thing mechanically work!?
      # Positional Encoding
      x = rearrange(x, 'b t w e -> (b t) w e')
      x = x + self.pos_embed

      # Temporal Encoding
      x = rearrange(x, '(b t) w e -> (b w) t e', b=B, t=T, w=W)
      x = x + self.time_embed
      x = rearrange(x, '(b w) t e -> b t w e', b=B, t=T, w=W)

      for blk in self.blocks:
        x = blk(x, B, T, W)
      
      return x

  def generationHead(self, x):
    B,T,W,E = x.shape
    # B T W E  -> (B T W) E
    x = rearrange(x, 'b t w e -> (b t w) e')
    x = self.decoder(x)
    # (B T W) 3 h w -> B T W 3 h w
    x = rearrange(x, '(b t k) c h w -> b t k c h w',b=B,t=T,k=W)
    # B T W[:-1] 3 h w
    x = x[:,:,:-1,:,:,:]
    # B T 3 ImgSize, ImgSize
    x = x.view(B,T,3,self.img_size,self.img_size)
    return x

  def forward(self, x, cont):
      x = self.forward_features(x, cont)
      x = self.generationHead(x)
      return x