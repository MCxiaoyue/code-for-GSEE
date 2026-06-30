import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
# 注意：确保 utils.py 在路径中，或者替换为你自己的 init_weights 和 get_padding 实现
from utils import init_weights, get_padding
import math

LRELU_SLOPE = 0.1


# ==========================================
# Backbone Components (EVRNet Parts)
# ==========================================

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        # 使用 Conv2d 处理 (Time, Channel) 维度，kernel=(k, 1) 表示只在时间轴卷积
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.act(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        # kernel=(1, k) 表示只在频率/通道轴卷积
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.act(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y0 = self.conv1(x)
        fused = y0 + x
        fused = self.act(fused)
        y1 = self.conv2(fused)
        output = y1 + fused
        return self.act(output)


class EVRNet_Backbone(nn.Module):
    def __init__(self):
        super(EVRNet_Backbone, self).__init__()
        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)
        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)
        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)

    def forward(self, x):
        # 输入 x: (B, Time, Channels)
        # 增加通道维度变为 (B, 1, Time, Channels) 以适配 Conv2d
        x = x.unsqueeze(1)

        x = self.spatial_block1(x)  # (B, 32, T/2, C)
        x = self.mkrb1(x)
        x = self.temporal_block1(x)  # (B, 32, T/2, C/2)
        x = self.mkrb2(x)
        x = self.temporal_block3(x)  # (B, 32, T/4, C/4)

        return x


# ==========================================
# Generator Components
# ==========================================

class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=dilation[0],
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=dilation[1],
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=dilation[2],
                       padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class EVRNet_Generator(torch.nn.Module):
    def __init__(self, h):
        super(EVRNet_Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1

        # ==========================================
        # 1. 集成 Backbone
        # ==========================================
        self.backbone = EVRNet_Backbone()
        backbone_out_channels = 32

        # ==========================================
        # 【新增】2. 强制下采样层 (为了匹配 130 长度)
        # ==========================================
        # 473 / 4 ≈ 118 (接近 130)
        # 我们使用 Kernel=5, Stride=4, Padding=2 来近似 1/4 的压缩
        self.conv_downsample = weight_norm(
            Conv1d(backbone_out_channels, backbone_out_channels,
                   kernel_size=5, stride=4, padding=2)
        )
        self.conv_downsample.apply(init_weights)

        # ==========================================
        # 3. 原有投影层 (输入通道保持不变)
        # ==========================================
        self.conv_backbone_proj = weight_norm(
            Conv1d(backbone_out_channels, h.in_ch, kernel_size=1, stride=1, padding=0)
        )

        # ==========================================
        # 4. 原有生成器结构
        # ==========================================
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch,
                   h.ch_init_upsample // 2,
                   3, 1,
                   padding=get_padding(3, 1)))

        self.GRU = nn.GRU(h.ch_init_upsample // 2,
                          h.ch_init_upsample // 4,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates,
                                       h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.ch_init_upsample // (2 ** i),
                                h.ch_init_upsample // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.conv_mid1 = weight_norm(
            Conv1d(h.ch_init_upsample // (2 ** self.i_mid),
                   h.ch_init_upsample // (2 ** self.i_mid),
                   3, 1,
                   padding=0))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.ch_init_upsample // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes,
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(ch,
                   h.out_ch,
                   9, 1,
                   padding=get_padding(9, 1)))

        # 初始化权重
        self.conv_backbone_proj.apply(init_weights)
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)

    def forward(self, x):
        # ==========================================
        # Step 1: Backbone 处理
        # ==========================================
        x_backbone = self.backbone(x)

        B, C, T, F1 = x_backbone.shape

        # 展平 (Batch, Channels, Time*Freq)
        x = x_backbone.permute(0, 1, 2, 3).contiguous().view(B, C, T * F1)

        # ==========================================
        # Step 2: 强制下采样 (Stride 4)
        # ==========================================
        # 这里会将长度从 473 压缩到 ~118
        x = self.conv_downsample(x)

        # ==========================================
        # Step 3: 投影通道数
        # ==========================================
        x = self.conv_backbone_proj(x)

        # ==========================================
        # Step 4: 原有生成器流程
        # ==========================================
        x = self.conv_pre(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        for i in range(self.num_upsamples):
            if i == self.i_mid:
                x = self.conv_mid1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        # 此时输出长度应为 118 左右，非常接近 130，且未使用插值
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_mid1)
        remove_weight_norm(self.conv_backbone_proj)
        remove_weight_norm(self.conv_downsample)  # 记得移除这里的权重范数


# ==========================================
# Discriminator (保持不变)
# ==========================================

class Discriminator(torch.nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.h = h
        self.ch_init_downsample = h.ch_init_downsample
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.n_classes = h.n_classes
        self.input_size = h.input_size
        self.m = 1

        for j in range(len(h.downsample_rates)):
            self.m = self.m * h.downsample_rates[j]

        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch,
                   h.ch_init_downsample,
                   3, 1,
                   padding=get_padding(3, 1)))

        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates,
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample * (2 ** i),
                       h.ch_init_downsample * (2 ** (i + 1)),
                       k, u, padding=math.ceil((k - u) / 2))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample * (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes,
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.GRU = nn.GRU(ch, ch // 2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9, 1)))

        # FC Layer
        self.adv_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), 1),
            nn.Sigmoid())
        self.aux_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), h.n_classes),
            nn.Softmax(dim=1))

        # 【新增】计算模型期望的下采样后时间长度
        expected_feature_dim = h.ch_init_downsample * 2 * 8 * (self.input_size // self.m)
        self.expected_time_len = expected_feature_dim // (h.ch_init_downsample * 2 * 8)

        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        # 使用自适应平均池化
        x = F.adaptive_avg_pool1d(x, self.expected_time_len)

        # FC Layer
        x = x.view(-1,
                   self.ch_init_downsample
                   * 2 * 8 * (self.input_size // self.m))
        validity = self.adv_classifier(x)
        label = self.aux_classifier(x)

        return validity, label

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)