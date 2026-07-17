import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init



class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size=7
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )


        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)


        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)


        return x


class RegionAdaptiveKANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        # 两个专家
        self.layer_lesion = KANLayer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, no_kan=no_kan
        )
        self.layer_bg = KANLayer(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, no_kan=no_kan
        )


        # 区域门控 conv
        mid_ch = max(dim // 4, 4)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        x: [B, N, C], N = H*W
        """
        B, N, C = x.shape

        x_norm = self.norm2(x)  # [B,N,C]

        # 还原到 2D 做空间门控
        x_2d = x_norm.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]
        gate = torch.sigmoid(self.gate_conv(x_2d))         # [B,1,H,W]
        gate_flat = gate.flatten(2).transpose(1, 2)        # [B,N,1]

        # 双专家映射
        y_les = self.layer_lesion(x_norm, H, W)            # [B,N,C]
        y_bg  = self.layer_bg(x_norm, H, W)                # [B,N,C]

        y_mixed = gate_flat * y_les + (1.0 - gate_flat) * y_bg

        out = x + self.drop_path(y_mixed)
        return out




class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class SemanticGatedSkip(nn.Module):
    def __init__(self, in_ch_enc, in_ch_dec, reduction=4, kernel_size=7):
        super().__init__()
        # 语义对齐通道
        mid_ch = max(in_ch_enc // reduction, in_ch_dec // reduction, 8)
        self.enc_proj = nn.Conv2d(in_ch_enc, mid_ch, kernel_size=1, bias=True)
        self.dec_proj = nn.Conv2d(in_ch_dec, mid_ch, kernel_size=1, bias=True)

        # 通道注意力：映射回 encoder 通道数
        hidden_ch = max(mid_ch // reduction, 4)
        self.mlp1 = nn.Linear(mid_ch, hidden_ch, bias=True)
        self.mlp2 = nn.Linear(hidden_ch, in_ch_enc, bias=True)

        # 空间注意力
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=True
        )

        # 融合后的卷积（输出保持 decoder 通道数）
        self.fuse_conv = nn.Conv2d(
            in_ch_enc + in_ch_dec, in_ch_dec, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x_enc, x_dec):
        """
        x_enc: encoder skip 特征 [B, C_e, H, W]
        x_dec: decoder 当前特征 [B, C_d, H, W]
        返回: 融合后的 decoder 特征 [B, C_d, H, W]
        """
        # 1) 语义对齐
        q = self.enc_proj(x_enc)              # [B, C_m, H, W]
        g = self.dec_proj(x_dec)              # [B, C_m, H, W]
        s = F.relu(q + g, inplace=True)       # 语义融合

        B, C_m, H, W = s.shape

        # 2) 通道注意力（针对 encoder 通道）
        z = F.adaptive_avg_pool2d(s, 1).view(B, C_m)     # [B, C_m]
        z = F.relu(self.mlp1(z), inplace=True)           # [B, C_m/r]
        u = torch.sigmoid(self.mlp2(z)).view(B, -1, 1, 1)  # [B, C_e, 1, 1]

        # 3) 空间注意力
        avg_pool = torch.mean(s, dim=1, keepdim=True)     # [B, 1, H, W]
        max_pool, _ = torch.max(s, dim=1, keepdim=True)   # [B, 1, H, W]
        spatial = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        a_s = torch.sigmoid(self.spatial_conv(spatial))   # [B, 1, H, W]

        # 4) 联合门控 + skip 融合
        a = u * a_s                                       # 广播到 [B, C_e, H, W]
        x_enc_gated = a * x_enc                           # 过滤后的 encoder 特征

        y = torch.cat([x_dec, x_enc_gated], dim=1)        # 拼接 [B, C_d + C_e, H, W]
        y = self.fuse_conv(y)                             # [B, C_d, H, W]

        return y




class BPNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,
                img_size=224, patch_size=16, in_chans=3,
                embed_dims=[256, 320, 512], no_kan=False,
                disease_classes=4,
                drop_rate=0., drop_path_rate=0.,
                norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        assert num_classes == 1, "当前版本 BPNet 只做二分类输出，num_classes 必须为 1。"
        assert disease_classes >= 1, "disease_classes 至少为 1。"

        kan_input_dim = embed_dims[0]

        # ========== Encoder ==========
        self.encoder1 = ConvLayer(input_channels, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ========== Tokenized KAN Stage ==========

        self.block1 = nn.ModuleList([
            RegionAdaptiveKANBlock(
                dim=embed_dims[1],
                drop=drop_rate,
                drop_path=dpr[0],
                norm_layer=norm_layer
            )
        ])

        self.block2 = nn.ModuleList([
            RegionAdaptiveKANBlock(
                dim=embed_dims[2],
                drop=drop_rate,
                drop_path=dpr[1],
                norm_layer=norm_layer
            )
        ])

        self.dblock1 = nn.ModuleList([
            RegionAdaptiveKANBlock(
                dim=embed_dims[1],
                drop=drop_rate,
                drop_path=dpr[0],
                norm_layer=norm_layer
            )
        ])

        self.dblock2 = nn.ModuleList([
            RegionAdaptiveKANBlock(
                dim=embed_dims[0],
                drop=drop_rate,
                drop_path=dpr[1],
                norm_layer=norm_layer
            )
        ])


        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1]
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2]
        )

        # ========== Decoder ==========
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        C_last = embed_dims[0] // 8  # 最后一层特征通道数


        self.seg_head = nn.Conv2d(C_last, 1, kernel_size=1)
        self.edge_head = nn.Conv2d(C_last, 1, kernel_size=1)


        self.prototype_lesion = nn.Parameter(torch.randn(disease_classes, C_last))
        self.prototype_bg = nn.Parameter(torch.randn(C_last))



    def forward(self, x):
        B = x.shape[0]

        # ========== Encoder ==========
        # Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        # Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        # Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        # ========== Tokenized KAN Stage ==========
        # Stage 4 (encoder KAN)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # ========== Decoder ==========
        # Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t4
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        # Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t3
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Stage 2 & 1
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t2
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = out + t1

        feat = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        # feat: [B, C_last, H, W]  —— 用于分割、边界和原型约束

        logits = self.seg_head(feat)    # [B,1,H,W]  主分割 (logits)
        edge_pred = self.edge_head(feat)  # [B,1,H,W] 边界预测 (logits)

        # 训练时用三个输出；推理时只用 logits 即可
        return logits, edge_pred, feat
