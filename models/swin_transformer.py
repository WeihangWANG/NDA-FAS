# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .pre_net import Pre_Net
from .mixer import Mixer

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    # def __init__(self, num_classes=2, **kwargs):
    def __init__(self, num_classes=2):
        super().__init__()

        self.pre = Pre_Net(BasicBlock)
        self.mixer = Mixer(BasicBlock)

        self.clr_fc = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(512, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 2))

        # self.mlp = Mlp(in_features=2*2*640, hidden_features=2*640, out_features=num_classes)

        # self.norm_mlp = norm_layer(2*2*640)
        

    # def image_into_patches(self, image, target_patch=32, overlap=0.25):
    #     # print("*****image shape*******")
    #     # print(image.shape)
    #     b, c, width, height = image.shape
    #     overlap_size = int(target_patch * (1 - overlap))

    #     patch_num = (width - target_patch) // overlap_size + 1

    #     starts = []
    #     for i in range(patch_num):
    #         for j in range(patch_num):
    #             starts.append([overlap_size * i, overlap_size * j])

    #     images = []
    #     for start_index in starts:
    #         x, y = start_index

    #         if x < 0:
    #             x = 0
    #         if y < 0:
    #             y = 0

    #         if x + target_patch >= width:
    #             x = width - target_patch - 1
    #         if y + target_patch >= height:
    #             y = height - target_patch - 1

    #         patch = image[:,:,x:x + target_patch, y: y + target_patch]
    #         images.append(patch)

    #     patches = torch.stack(images,dim=1)
    #     # print("*********patches shape**********")
    #     # print(patches.shape)

    #     return patches

    def forward(self, x):
        b, n, c, h, w = x.shape

        fea = self.pre.forward(x)
        fea = self.mixer.forward(fea)

        fea = torch.mean(fea, dim=2)
        fea_mean = F.adaptive_avg_pool2d(fea, output_size=1).view(b, -1)
        fea_mean = F.normalize(fea_mean, p=1, dim=1)
        res = self.clr_fc(fea_mean)

        return res, fea_mean
        # return fea_mean
        # x_p = self.image_into_patches(x)
        # b, n, c, h, w = x_p.shape

        # fea = self.pre.forward(x_p)
        # fea = self.mixer.forward(fea)

        # fea = torch.mean(fea, dim=2)

        # fea_mean = torch.flatten(fea,1)
        # fea_mean = self.norm_mlp(fea_mean)
        # # fea_mean = F.adaptive_avg_pool2d(fea, output_size=1).view(b, -1)
        # # res = self.clr_fc(fea_mean)
        # res = self.mlp(fea_mean)
        # return res

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
