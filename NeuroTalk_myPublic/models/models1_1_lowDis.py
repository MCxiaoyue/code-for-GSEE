import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
# 注意：确保 utils.py 在路径中，或者替换为你自己的 init_weights 和 get_padding 实现
from utils import init_weights, get_padding
import math
from mamba.mamba import Mamba

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
        # self.i_mid = 0
        # self.i_mid_gru = 1

        # --- 1. 完整复刻分类器的结构 (为了加载权重) ---
        self.backbone = EVRNet_Backbone()
        # 注意：这里必须和分类器一样，后面接 AvgPool 和 Dropout
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)

        # Mamba 定义 (必须和分类器完全一致)
        # 假设分类器里 Mamba 输入是 32 (因为 AvgPool 后是 1x1)
        self.mamba = Mamba(num_layers=1, d_input=32, d_model=8, d_state=8, d_discr=16, ker_size=4,
                           parallel=False)

        # --- 2. 【新增】特征还原/扩展层 ---
        # Mamba 输出是 (B, 1, 16) 这种极小的向量
        # 我们需要把它变成生成器能用的时序特征
        # 假设我们要把它扩展回 (B, Channels, Time)

        d_discr = 32  # 对应 Mamba 的 d_input
        target_ch = 1024  # 假设生成器后续需要的通道数

        # 这个卷积层负责把 Mamba 的 16维向量 映射到 生成器的通道数
        # 此时 Time=1，我们后面靠上采样来拉长时间
        self.mamba_to_gen_proj = weight_norm(
            Conv1d(d_discr, target_ch, kernel_size=1)
        )

        # 2. 【新增】序列长度扩展层
        # 我们需要把长度从 1 变成比如 15 (假设总上采样倍率是 8x8=64, 15*64=960)
        # 或者我们可以直接在这里用 Interpolate 把长度拉长
        # 这里我们用一个简单的线性插值或者重复策略

        # 方案 A: 使用插值层 (推荐)
        # 假设我们希望进入第一个 Upsample 层之前，长度至少是 16
        self.mamba_seq_len_upsample = 16
        self.upsample_len = torch.nn.Upsample(size=self.mamba_seq_len_upsample, mode='linear', align_corners=False)

        # ==========================================
        # 3. 原有投影层 (输入通道保持不变)
        # ==========================================

        # ==========================================
        # 4. 原有生成器结构
        # ==========================================

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates,
                                       h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.ch_init_upsample // (2 ** i),
                                h.ch_init_upsample // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.conv_mid1 = weight_norm(
            Conv1d(1024,
                   1024,
                   3, 1,
                   padding=1))

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
        # self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)

    def forward(self, x):
        # --- 1. Backbone (同分类器) ---
        x = self.backbone(x)  # (B, 32, T, F)

        # --- 2. AvgPool (同分类器 - 这一步会丢失时序信息，但为了权重必须做) ---
        x = self.avg_pooling(x)  # (B, 32, 1, 1)

        # --- 3. 准备 Mamba 输入 (同分类器) ---
        x = x.view(x.size(0), -1)  # (B, 32)
        x = self.dropout(x)

        # --- 4. Mamba (同分类器) ---

        x = x.unsqueeze(-2)  # (B, 1, 32)
        x = self.mamba(x)[0]  # (B, 1, 16) -> 这里的 1 是序列长度


        x = x.permute(0, 2, 1)  # (B, 16, 1)
        x = self.mamba_to_gen_proj(x)  # (B, target_ch, 1)

        # ==========================================
        # --- 5. 【关键修改】手动扩展序列长度 ---
        # ==========================================
        # 将长度从 1 扩展到 16 (或者你设定的其他值)
        # 这样后续的 Conv1d(kernel_size=3) 就可以正常工作了
        x = self.upsample_len(x)  # (B, 1024, 16)

        # --- 6. 中间卷积 (现在安全了) ---
        x = self.conv_mid1(x)  # (B, 1024, 16)

        # ==========================================
        # Step 4: 原有生成器流程
        # ==========================================
        for i in range(self.num_upsamples):
            # if i == self.i_mid:
            #     x = self.conv_mid1(x)
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

# ==========================================
# Discriminator (已修改：适应简化的配置)
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

        # 动态计算总的下采样倍率
        self.m = 1
        for rate in h.downsample_rates:
            self.m *= rate

        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch,
                   h.ch_init_downsample,
                   3, 1,
                   padding=get_padding(3, 1)))

        # 创建下采样层
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates,
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample * (2 ** i),
                       h.ch_init_downsample * (2 ** (i + 1)),
                       k, u, padding=math.ceil((k - u) / 2))))

        # 创建 ResBlocks (兼容旧的配置文件格式)
        self.resblocks = nn.ModuleList()
        for i in range(self.num_downsamples):
            ch = h.ch_init_downsample * (2 ** (i + 1))  # 通道数随层数翻倍
            # 为当前下采样层创建 num_kernels 个 ResBlock
            for j in range(self.num_kernels):
                # 为兼容旧格式，我们使用提供的 resblock_dilation_sizes[i] 来为每个kernel设置dilation
                current_dilation_config = h.resblock_dilation_sizes[i]
                # 创建 ResBlock，传入正确的通道数和 dilation 配置
                self.resblocks.append(ResBlock(h, ch, h.resblock_kernel_sizes[j], current_dilation_config))

        # GRU 输入的通道数是最后一层下采样后的通道数
        gru_input_ch = h.ch_init_downsample * (2 ** self.num_downsamples)
        self.GRU = nn.GRU(gru_input_ch, gru_input_ch // 2,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

        # GRU 输出后，特征会被 concat，通道数翻倍
        final_channels_before_pool = gru_input_ch * 2  # 例如，如果 num_downsamples=2, 最终通道是 256, concat后是 512

        # 为了计算 FC 层的输入特征数，我们使用自适应池化来固定长度
        # 假设我们希望池化后的长度是基于原始配置的一个估算值
        original_formula_result = h.ch_init_downsample * 2 * 8 * (self.input_size // self.m)
        self.pooled_length = max(1, math.ceil(original_formula_result / final_channels_before_pool))
        self.fc_input_features = final_channels_before_pool * self.pooled_length

        self.adv_classifier = nn.Sequential(nn.Linear(self.fc_input_features, 1), nn.Sigmoid())
        self.aux_classifier = nn.Sequential(nn.Linear(self.fc_input_features, h.n_classes), nn.Softmax(dim=1))

        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        # 移除了 self.conv_post 的定义和初始化

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                global_index = i * self.num_kernels + j
                resblock_output = self.resblocks[global_index](x)

                if xs is None:
                    xs = resblock_output
                else:
                    xs += resblock_output
            x = xs / self.num_kernels

        x = F.leaky_relu(x)

        # GRU 部分处理到最后一个下采样层的输出
        x_gru_in = x  # Shape: (B, final_ch, T_final)
        x_gru_in = x_gru_in.transpose(1, 2)  # Shape: (B, T_final, final_ch)
        self.GRU.flatten_parameters()
        x_gru_out, _ = self.GRU(x_gru_in)  # Shape: (B, T_final, final_ch)
        x_gru_out = x_gru_out.transpose(1, 2)  # Shape: (B, final_ch, T_final)

        # Concatenate residual connection
        x = torch.cat([x_gru_out, x], dim=1)  # Shape: (B, final_ch * 2, T_final)

        # 使用自适应平均池化，将时间维度池化到 self.pooled_length
        x = F.adaptive_avg_pool1d(x, self.pooled_length)  # Shape: (B, final_ch * 2, self.pooled_length)

        # FC Layer: Flatten the features
        x = x.view(-1, self.fc_input_features)  # Now it should match
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
        # 已移除对不存在的 'conv_post' 的操作


# class Discriminator(torch.nn.Module):
#     def __init__(self, h):
#         super(Discriminator, self).__init__()
#         self.h = h
#         self.ch_init_downsample = h.ch_init_downsample
#         self.num_kernels = len(h.resblock_kernel_sizes)
#         self.num_downsamples = len(h.downsample_rates)
#         self.n_classes = h.n_classes
#         self.input_size = h.input_size
#         self.m = 1
#
#         for j in range(len(h.downsample_rates)):
#             self.m = self.m * h.downsample_rates[j]
#
#         # model define
#         self.conv_pre = weight_norm(
#             Conv1d(h.in_ch,
#                    h.ch_init_downsample,
#                    3, 1,
#                    padding=get_padding(3, 1)))
#
#         self.downs = nn.ModuleList()
#         for i, (u, k) in enumerate(zip(h.downsample_rates,
#                                        h.downsample_kernel_sizes)):
#             self.downs.append(weight_norm(
#                 Conv1d(h.ch_init_downsample * (2 ** i),
#                        h.ch_init_downsample * (2 ** (i + 1)),
#                        k, u, padding=math.ceil((k - u) / 2))))
#
#         self.resblocks = nn.ModuleList()
#         for i in range(len(self.downs)):
#             ch = h.ch_init_downsample * (2 ** (i + 1))
#             for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes,
#                                            h.resblock_dilation_sizes)):
#                 self.resblocks.append(ResBlock(h, ch, k, d))
#
#         self.GRU = nn.GRU(ch, ch // 2,
#                           num_layers=1,
#                           batch_first=True,
#                           bidirectional=True)
#
#         self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9, 1)))
#
#         # FC Layer
#         self.adv_classifier = nn.Sequential(nn.Linear(
#             h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), 1),
#             nn.Sigmoid())
#         self.aux_classifier = nn.Sequential(nn.Linear(
#             h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), h.n_classes),
#             nn.Softmax(dim=1))
#
#         # 【新增】计算模型期望的下采样后时间长度
#         expected_feature_dim = h.ch_init_downsample * 2 * 8 * (self.input_size // self.m)
#         self.expected_time_len = expected_feature_dim // (h.ch_init_downsample * 2 * 8)
#
#         self.conv_pre.apply(init_weights)
#         self.downs.apply(init_weights)
#         self.conv_post.apply(init_weights)
#
#     def forward(self, x):
#         x = self.conv_pre(x)
#
#         for i in range(self.num_downsamples):
#             x = F.leaky_relu(x, LRELU_SLOPE)
#             x = self.downs[i](x)
#
#             xs = None
#             for j in range(self.num_kernels):
#                 if xs is None:
#                     xs = self.resblocks[i * self.num_kernels + j](x)
#                 else:
#                     xs += self.resblocks[i * self.num_kernels + j](x)
#             x = xs / self.num_kernels
#         x = F.leaky_relu(x)
#         x_temp = x
#         x = x.transpose(1, 2)
#         self.GRU.flatten_parameters()
#         x, _ = self.GRU(x)
#         x = x.transpose(1, 2)
#         x = torch.cat([x, x_temp], dim=1)
#
#         # 使用自适应平均池化
#         x = F.adaptive_avg_pool1d(x, self.expected_time_len)
#
#         # FC Layer
#         x = x.view(-1,
#                    self.ch_init_downsample
#                    * 2 * 8 * (self.input_size // self.m))
#         validity = self.adv_classifier(x)
#         label = self.aux_classifier(x)
#
#         return validity, label
#
#     def remove_weight_norm(self):
#         print('Removing weight norm...')
#         for l in self.downs:
#             remove_weight_norm(l)
#         for l in self.resblocks:
#             l.remove_weight_norm()
#         remove_weight_norm(self.conv_pre)
#         remove_weight_norm(self.conv_post)