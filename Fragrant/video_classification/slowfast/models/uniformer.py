from math import ceil, sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .build import MODEL_REGISTRY
import os
import torch.fft as fft
import einops
from einops import rearrange
import numbers

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

model_path = 'path_to_models'
model_path = {
    'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
}


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim):
    return nn.BatchNorm3d(dim)


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


@MODEL_REGISTRY.register()
class Uniformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, cfg):
        super().__init__()

        depth = cfg.UNIFORMER.DEPTH
        num_classes = cfg.MODEL.NUM_CLASSES
        img_size = cfg.DATA.TRAIN_CROP_SIZE
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        embed_dim = cfg.UNIFORMER.EMBED_DIM
        head_dim = cfg.UNIFORMER.HEAD_DIM
        mlp_ratio = cfg.UNIFORMER.MLP_RATIO
        qkv_bias = cfg.UNIFORMER.QKV_BIAS
        qk_scale = cfg.UNIFORMER.QKV_SCALE
        representation_size = cfg.UNIFORMER.REPRESENTATION_SIZE
        drop_rate = cfg.UNIFORMER.DROPOUT_RATE
        attn_drop_rate = cfg.UNIFORMER.ATTENTION_DROPOUT_RATE
        drop_path_rate = cfg.UNIFORMER.DROP_DEPTH_RATE
        split = cfg.UNIFORMER.SPLIT
        std = cfg.UNIFORMER.STD
        self.use_checkpoint = cfg.MODEL.USE_CHECKPOINT
        self.checkpoint_num = cfg.MODEL.CHECKPOINT_NUM

        logger.info(f'Use checkpoint: {self.use_checkpoint}')
        logger.info(f'Checkpoint number: {self.checkpoint_num}')

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer)
                for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer)
                for i in range(depth[2])])
            self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer)
                for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.disentangle = DisEntangle()
        # self.multi_attn = NextAttentionZ(embed_dim[1], head_dim)
        self.rsa_attn1 = RSA(64, 64)
        self.rsa_attn2 = RSA(128, 128)
        self.rsa_attn3 = RSA(320, 320)
        self.rsa_attn4 = RSA(512, 512)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, cfg):
        if cfg.UNIFORMER.PRETRAIN_NAME:
            # checkpoint = torch.load(model_path[cfg.UNIFORMER.PRETRAIN_NAME], map_location='cpu')
            checkpoint = torch.load(
                "/public/home/liuwx/perl5/UniFormerOriginal/video_classification/exp/uniformer_s8x8_k400/uniformer_small_in1k.pth",
                map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        logger.info(f'Ignore: {k}')
                        continue
                    logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        # print(se)
        x = self.patch_embed1(x)
        # ####################################################
        # x += self.rsa_attn1(x)
        # x = x + self.disentangle(x)
        # ####################################################
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed2(x)

        # ####################################################
        # x += self.rsa_attn2(x)
        # x = x + self.disentangle(x)
        # ####################################################

        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed3(x)

        # ####################################################
        # x += self.rsa_attn3(x)
        # x = x + self.disentangle(x)
        # ####################################################

        # x = self.disentangle(x)
        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_embed4(x)
        # ####################################################
        # x = x + self.disentangle(x)
        # x += self.rsa_attn4(x)
        # ####################################################

        for i, blk in enumerate(self.blocks4):
            if self.use_checkpoint and i < self.checkpoint_num[3]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)

        return x

    def forward(self, x):
        x = x[0]
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        feat = x
        x = self.head(x)
        return x, feat


class DisEntangle(nn.Module):
    def __init__(self):
        super(DisEntangle, self).__init__()

    def forward(self, x):
        shape = x.shape
        frequencies = fft.fftfreq(shape[2]).cuda()
        fft_compute = fft.fft(x, dim=2, norm='ortho').abs()
        frequencies = frequencies.unsqueeze(1)
        frequencies = frequencies.unsqueeze(1)
        frequencies = frequencies.unsqueeze(0)
        frequencies = frequencies.unsqueeze(0)

        x = x * frequencies * frequencies * fft_compute
        # * fft_compute
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies.cpu().numpy(), x.cpu().numpy())
        plt.xlabel('Frequency')
        plt.ylabel('Weighted Energy')
        print(12334444)
        plt.title('Weighted Energy Distribution in Frequency Domain (After Weighting)')
        plt.savefig(f'weighted_energy_distribution_after_weighting_{i}_{j}.png')  # 保存图像
        plt.close()

        return x


class RSA(nn.Module):
    def __init__(
            self,
            d_in=128,
            d_out=128,
            nh=1,
            dk=16,
            dv=0,
            dd=0,
            kernel_size=(3, 7, 7),
            stride=(1, 1, 1),
            kernel_type='V',  # ['V', 'R', 'VplusR']
            feat_type='V',  # ['V', 'R', 'VplusR']
    ):
        super(RSA, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.nh = nh
        self.dv = dv = d_out // nh if dv == 0 else dv
        self.dk = dk = dv if dk == 0 else dk
        self.dd = dd = dk if dd == 0 else dd

        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_type = kernel_type
        self.feat_type = feat_type

        assert self.kernel_type in ['V', 'R', 'VplusR'], "Not implemented involution type: {}".format(self.kernel_type)
        assert self.feat_type in ['V', 'R', 'VplusR'], "Not implemented feature type: {}".format(self.feat_type)

        print("d_in: {}, d_out: {}, nh: {}, dk: {}, dv: {}, dd:{}, kernel_size: {}, kernel_type: {}, feat_type: {}"
              .format(d_in, d_out, nh, dk, dv, self.dd, kernel_size, kernel_type, feat_type))

        self.ksize = ksize = kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.pad = pad = tuple(k // 2 for k in kernel_size)

        # hidden dimension
        d_hid = nh * dk + dv if self.kernel_type == 'V' else nh * dk + dk + dv

        # Linear projection
        self.projection = nn.Conv3d(d_in, d_hid, 1, bias=False)

        # Intervolution Kernel
        if self.kernel_type == 'V':
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        elif self.kernel_type == 'R':
            self.H1 = nn.Conv3d(dk, dk * dd, kernel_size, padding=self.pad, groups=dk, bias=False)
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        elif self.kernel_type == 'VplusR':
            self.P1 = nn.Parameter(tr.randn(dk, dd).unsqueeze(0) * sqrt(1 / (ksize * dd)), requires_grad=True)
            self.H1 = nn.Conv3d(dk, dk * dd, kernel_size, padding=self.pad, groups=dk, bias=False)
            self.H2 = nn.Conv3d(1, dd, kernel_size, padding=self.pad, bias=False)
        else:
            raise NotImplementedError

            # Feature embedding layer
        if self.feat_type == 'V':
            pass
        elif self.feat_type == 'R':
            self.G = nn.Conv3d(1, dv, kernel_size, padding=self.pad, bias=False)
        elif self.feat_type == 'VplusR':
            self.G = nn.Conv3d(1, dv, kernel_size, padding=self.pad, bias=False)
            self.I = nn.Parameter(tr.eye(dk).unsqueeze(0), requires_grad=True)
        else:
            raise NotImplementedError

            # Downsampling layer
        if max(self.stride) > 1:
            self.avgpool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def L2norm(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, x):
        N, C, T, H, W = x.shape

        '''Linear projection'''
        x_proj = self.projection(x)

        if self.kernel_type != 'V':
            q, k, v = torch.split(x_proj, [self.nh * self.dk, self.dk, self.dv], dim=1)
        else:
            q, v = torch.split(x_proj, [self.nh * self.dk, self.dv], dim=1)

        '''Normalization'''
        q = rearrange(q, 'b (nh k) t h w -> b nh k t h w', k=self.dk)
        q = self.L2norm(q, d=2)
        q = rearrange(q, 'b nh k t h w -> (b t h w) nh k')

        v = self.L2norm(v, d=1)

        if self.kernel_type != 'V':
            k = self.L2norm(k, d=1)

        '''
        q = (b t h w) nh k
        k = b k t h w
        v = b v t h w
        '''

        # Intervolution generation
        # Basic kernel
        if self.kernel_type is 'V':
            kernel = q
        # Relational kernel
        else:
            K_H1 = self.H1(k)
            K_H1 = rearrange(K_H1, 'b (k d) t h w -> (b t h w) k d', k=self.dk)

            if self.kernel_type == 'VplusR':
                K_H1 = K_H1 + self.P1

            kernel = torch.einsum('abc,abd->acd', q.transpose(1, 2), K_H1)  # (bthw, nh, d)

        # feature generation
        # Appearance feature
        v = rearrange(v, 'b (v 1) t h w -> (b v) 1 t h w')

        V = self.H2(v)  # (bv, d, t, h, w)
        feature = rearrange(V, '(b v) d t h w -> (b t h w) v d', v=self.dv)

        # Relational feature
        if self.feat_type in ['R', 'VplusR']:
            V_G = self.G(v)  # (bv, v2, t, h, w)
            V_G = rearrange(V_G, '(b v) v2 t h w -> (b t h w) v v2', v=self.dv)

            if self.feat_type == 'VplusR':
                V_G = V_G + self.I

            feature = torch.einsum('abc,abd->acd', V_G, feature)  # (bthw, v2, d)

        # kernel * feat
        out = torch.einsum('abc,adc->adb', kernel, feature)  # (bthw, nh, v2)

        out = rearrange(out, '(b t h w) nh v -> b (nh v) t h w', t=T, h=H, w=W)

        if max(self.stride) > 1:
            out = self.avgpool(out)

        return out