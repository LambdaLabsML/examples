from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from torch import Tensor

from arch.gc_vit import GCViTLayer, PatchEmbed


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)
    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


class GlobalContextIR(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        depths: List[int],
        window_size,
        mlp_ratio,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        sr_scale: int = 2,
        **kwargs
    ) -> None:
        """
        Args:
            feature_dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super().__init__()

        self.sr_scale = sr_scale

        if kwargs['version'] == 1:
            num_features = feature_dim

            self.feature_dim = feature_dim
            self.num_classes = num_classes
            self.patch_embed = PatchEmbed(in_chans=in_chans, dim=feature_dim)
            self.pos_drop = nn.Dropout(p=drop_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            self.levels = nn.ModuleList()
            for i in range(len(depths)):
                level = GCViTLayer(dim=feature_dim,
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   norm_layer=norm_layer,
                                   downsample=False,
                                   layer_scale=layer_scale)
                self.levels.append(level)
            self.norm = norm_layer(num_features)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            # self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.head = nn.Conv2d(num_features, num_features, kernel_size=3)
            self.apply(self._init_weights)

            self.upsample_dim = 8
            self._init_upsampler(
                feature_dim=self.feature_dim,
                upsample_dim=self.upsample_dim,
                out_channels=3
            )
        elif kwargs['version'] == 2:
            self._build_version2(feature_dim, depths, window_size, mlp_ratio,
                                 num_heads, drop_path_rate, in_chans,
                                 num_classes, qkv_bias, qk_scale, drop_rate,
                                 attn_drop_rate, norm_layer, layer_scale, kwargs)

    def _build_version2(
        self,
        dim: int,
        depths: List[int],
        window_size,
        mlp_ratio,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        **kwargs
    ) -> None:
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Conv2d(num_features, num_features, kernel_size=3)

        self.upsample_dim = 64
        self._init_upsampler(
            feature_dim=self.feature_dim,
            upsample_dim=self.upsample_dim,
            out_channels=3
        )
        self.apply(self._init_weights)

    def _init_upsampler(
        self,
        feature_dim: int,
        upsample_dim: int,
        out_channels: int
    ) -> None:
        # if False == '1conv':
        #     self.conv_after_body = nn.Conv2d(upsample_dim, feature_dim, 3, 1, 1)
        # elif True == '3conv':
        # to save parameters and memory
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(upsample_dim, feature_dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim // 4, upsample_dim, 3, 1, 1)
        )

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(11, upsample_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_up1 = nn.Conv2d(upsample_dim, upsample_dim, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(upsample_dim, upsample_dim, 3, 1, 1)
        self.conv_hr = nn.Conv2d(upsample_dim, upsample_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(upsample_dim, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def upsample(self, init_x: Tensor, x: Tensor) -> Tensor:
        x = torch.cat((self.conv_after_body(x), init_x), dim=1)
        x = self.conv_before_upsample(x)
        x = self.lrelu(
            self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest'))
        )
        x = self.lrelu(self.conv_up2(x))
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        # print("AFTER PATCH EMBED: ", x.shape)
        x = self.pos_drop(x)
        # print("AFTER pos drop: ", x.shape)

        for level in self.levels:
            x = level(x)
            # print("AFTER level: ", x.shape)
        # print("AFTER levels: ", x.shape)

        x = self.norm(x)
        # x = _to_channel_first(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        # print("INITIAL x shape: ", x.shape)
        B, C, H, W = x.shape
        init_x = x

        x = self.forward_features(x)
        # print("AFTER forward features: ", x.shape)

        # scale_up = self.feature_dim // self.upsample_dim // 2
        # print("SCALE UP for reshape: ", scale_up)

        x = x.reshape(B, self.upsample_dim, H, W)
        # x = self.head(x)

        # print("AFTER reshape: ", x.shape)

        x = self.upsample(init_x, x)

        # print("AFTER upsample: ", x.shape)

        x = x.reshape(B, C, H * self.sr_scale, W * self.sr_scale)
        return x


@register_model
def gcir_nano(pretrained=False, **kwargs):
    model = GlobalContextIR(
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        # window_size=[7, 7, 14, 7],
        window_size=[8, 8, 8, 8],
        feature_dim=128,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_micro(pretrained=False, **kwargs):
    model = GlobalContextIR(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7, 14, 7],
        feature_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_tiny(pretrained=False, **kwargs):
    model = GlobalContextIR(
        depths=[3, 4, 19, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7, 14, 7],
        feature_dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_small(pretrained=False, **kwargs):
    model = GlobalContextIR(
        depths=[3, 4, 19, 5],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7, 14, 7],
        feature_dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_swinir_comp(pretrained=False, **kwargs):
    model = GlobalContextIR(
        # depths=[3, 4, 19, 5],
        # num_heads=[4, 8, 16, 32],
        depths=[6, 6, 6, 6],
        num_heads=[2, 4, 8, 16],
        # window_size=[7, 7, 14, 7],
        window_size=[8, 8, 8, 8],
        feature_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_regular(pretrained=False, **kwargs):
    model = GlobalContextIR(
        # depths=[3, 4, 19, 5],
        # num_heads=[4, 8, 16, 32],
        depths=[6, 6, 6, 6, 12, 12],
        num_heads=[2, 4, 8, 16, 32, 64],
        # window_size=[7, 7, 14, 7],
        window_size=[8, 8, 8, 8, 8, 8],
        feature_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def gcir_base(pretrained=False, **kwargs):
    model = GlobalContextIR(
        # depths=[3, 4, 19, 5],
        # num_heads=[4, 8, 16, 32],
        depths=[6, 6, 12, 12, 18, 18],
        num_heads=[2, 4, 8, 16, 32, 64],
        # window_size=[7, 7, 14, 7],
        window_size=[8, 8, 8, 8, 8, 8],
        feature_dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model
