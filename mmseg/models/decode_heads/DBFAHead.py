# Copyright (c) OpenMMLab. All rights reserved.
#######################
#### BIFPN + DYSAMPLE
########################3
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize, Upsample
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.dysample import DySample
from mmseg.models.decode_heads.bifpn.util import DepthWiseConvBlock, MaxPool2dSamePadding
from mmcv.cnn.bricks import Swish

#2024.1.12
class Mlp(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        embed_dim=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or input_dim
        hidden_dim = hidden_dim or input_dim*3 #2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BiFPNStage(nn.Module):
    '''
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        num_outs: int, BiFPN need feature maps num
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        epsilon: float, hyperparameter in fusion features
    '''

    def __init__(self, out_channels, apply_bn_for_resampling=True,
                 conv_bn_act_pattern=False,
                 norm_cfg=dict(type='BN', momentum=1e-2, eps=1e-3),
                 epsilon=1e-4,
                 style='lp',
                 dyscope=False) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.norm_cfg = norm_cfg
        self.epsilon = epsilon

        # self.p3_upsample = Upsample(scale_factor=2, mode='bilinear')
        # self.p4_upsample = Upsample(scale_factor=2, mode='bilinear')
        self.p3_upsample = DySample(in_channels=out_channels,scale=2, style=style, groups=4, dyscope=dyscope)
        self.p4_upsample = DySample(in_channels=out_channels,scale=2, style=style, groups=4, dyscope=dyscope)

        # bottom to up: feature map down_sample module
        self.p3_down_sample = MaxPool2dSamePadding(3, 2)
        self.p4_down_sample = MaxPool2dSamePadding(3, 2)

        # Fuse Conv Layers
        self.conv3_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv2_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)

        self.conv3_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv4_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)

        # weights
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()

        self.swish = Swish()

    def combine(self, x):
        if not self.conv_bn_act_pattern:
            x = self.swish(x)
        return x

    def forward(self, x2,x3,x4):
        p2_in, p3_in, p4_in = x2, x3, x4 #[2,768，64，64],[2,768，32，32],[2,768，16，16]

        # Weights for P4_0 and P5_1 to P4_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p3_up = self.conv3_up(
            self.combine(weight[0] * p3_in +
                         weight[1] * self.p4_upsample(p4_in)))  # (1,64,32,32)

        # Weights for P3_0 and P4_1 to P3_2
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p2_out = self.conv2_up(
            self.combine(weight[0] * p2_in +
                         weight[1] * self.p3_upsample(p3_up)))  # (1,64,64,64)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p3_out = self.conv3_down(
            self.combine(weight[0] * p3_in + weight[1] * p3_up +
                         weight[2] * self.p3_down_sample(p2_out)))  # (1,64,32,32)

        # Weights for P7_0 and P6_2 to P7_2
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p4_out = self.conv4_down(
            self.combine(weight[0] * p4_in +
                         weight[1] * self.p4_down_sample(p3_out)))  # (1,64,4,4)
        return p2_out, p3_out, p4_out


@HEADS.register_module(force=True)
class BifpnHead2(BaseDecodeHead):
    def __init__(self, style='lp', dyscope=False, **kwargs):
        super(BifpnHead2, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.linear_c4 = Mlp(input_dim=self.in_channels[3], embed_dim=self.channels)  # input_dim=self.channels
        self.linear_c3 = Mlp(input_dim=self.in_channels[2], embed_dim=self.channels)  # input_dim=self.channels
        self.linear_c2 = Mlp(input_dim=self.in_channels[1], embed_dim=self.channels)  # input_dim=self.channels

        self.bifpn = BiFPNStage(out_channels=self.channels,
                                apply_bn_for_resampling=True,
                                conv_bn_act_pattern=False,
                                norm_cfg=self.norm_cfg,
                                style=style,
                                dyscope=dyscope
                                )

        self.linear_c41 = Mlp(input_dim=self.channels, embed_dim=self.channels)
        self.linear_c31 = Mlp(input_dim=self.channels, embed_dim=self.channels)
        self.linear_c21 = Mlp(input_dim=self.channels, embed_dim=self.channels)
        self.linear_c1 = Mlp(input_dim=self.in_channels[0], embed_dim=48)

        self.output_cat = ConvModule(
            in_channels=self.channels * 3,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.low_level_fuse = ConvModule(
            in_channels=self.channels + 48,  # 768+48
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.c4_upsample = DySample(in_channels=self.channels, scale=4, style=style, groups=4, dyscope=dyscope)
        self.c3_upsample = DySample(in_channels=self.channels, scale=2, style=style, groups=4, dyscope=dyscope)
        self.c2_upsample = DySample(in_channels=self.channels, scale=2, style=style, groups=4, dyscope=dyscope)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs  # c2,torch.Size([2, 128, 64, 64])
        n, _, h, w = c4.shape

        ###############2024.1.14
        c2_mlp = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c3_mlp = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        c4_mlp = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ######################3
        c2_mlp_bi, c3_mlp_bi, c4_mlp_bi = self.bifpn(c2_mlp, c3_mlp, c4_mlp)
        # c2_mlp_bi[2,512,64,64],c3_mlp_bi[2,512,32,32],c4_mlp_bi[2,512,16,16]

        # s = c2_mlp_bi.size()[2:]
        # c4_up = resize(c4_mlp_bi, size=s, mode='bilinear', align_corners=False)
        # c3_up = resize(c3_mlp_bi, size=s, mode='bilinear', align_corners=False)
        c4_up = self.c4_upsample(c4_mlp_bi)
        c3_up = self.c3_upsample(c3_mlp_bi)
        output = self.output_cat(torch.cat([c4_up, c3_up, c2_mlp_bi], dim=1))  # [2,512,128,128]

        c1_mlp = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        output = self.c2_upsample(output)
        # output = resize(output, size=c1.size()[2:], mode='bilinear', align_corners=False)
        output = self.low_level_fuse(torch.cat([output, c1_mlp], dim=1))

        output = self.cls_seg(output)

        return output


if __name__ == '__main__':
    import numpy as np
    x1 = torch.from_numpy(np.ones((2, 64, 128, 128))).float()
    x2 = torch.from_numpy(np.ones((2, 128, 64, 64))).float()
    x3 = torch.from_numpy(np.ones((2, 256, 32, 32))).float()
    x4 = torch.from_numpy(np.ones((2, 512, 16, 16))).float()
    # x = torch.from_numpy(x).float()
    x1 = x1.clone().detach()
    x2 = x2.clone().detach()
    x3 = x3.clone().detach()
    x4 = x4.clone().detach()
    x = [x1,x2,x3,x4]
    model = BifpnHead2(
        in_channels=[64,128,256,512],
        in_index=[0, 1, 2, 3],
        channels=768,
        # dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True))
    x = model.forward(x)
    print(model)
