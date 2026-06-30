import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
# 注意：确保 utils.py 在路径中，或者替换为你自己的 init_weights 和 get_padding 实现
from utils import init_weights, get_padding
import math
# 引入 Mamba 模块
from mamba.mamba import Mamba

LRELU_SLOPE = 0.1


# ==========================================
# Backbone Components (EVRNet Parts) - 保持不变
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
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
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


class HybridGenerator(torch.nn.Module):
    """
    混合生成器：结合了 EVRNet_Backbone 和原始 Generator 的结构。
    【修复点】加入了 Mamba 层，以对齐训练脚本的权重加载逻辑。
    """

    def __init__(self, h):
        super(HybridGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1

        # --- 1. 完整的 EVRNet Backbone ---
        self.backbone = EVRNet_Backbone()

        # --- 2. 【新增】Mamba 层 (参数需与训练脚本/EVRNet_Classifier_Old 保持一致) ---
        # 这里的参数 (d_input, d_model 等) 必须与 EVRNet_Classifier_Old 中定义的完全相同
        self.mamba = Mamba(num_layers=1, d_input=32, d_model=8, d_state=8, d_discr=16, ker_size=4,
                           parallel=False)

        # 由于 Backbone 输出是 (B, 32, H, W)，我们需要将其映射到 Mamba 的输入维度
        # 假设我们在序列维度 (H*W) 上应用 Mamba，或者在通道维度上。
        # 根据分类器逻辑，通常是在序列长度上。这里我们假设在 flattened spatial-temporal 维度上操作。
        # 我们需要一个投影层将 Backbone 的 Channel (32) 映射到 Mamba 的 d_model (8)
        self.mamba_proj_in = nn.Linear(32, 32)  # 32 (backbone out) -> 8 (mamba d_model)
        self.mamba_proj_out = nn.Linear(32, 32)  # 8 (mamba out) -> 32 (back to gru input)

        # --- 3. 【关键】Backbone 到 GRU 的桥接层 (保持不变) ---
        backbone_out_channels = 32
        gru_input_dim = h.ch_init_upsample // 2  # 这是原始 Generator 中 GRU 的输入维度
        self.backbone_to_gru_proj = nn.Conv2d(backbone_out_channels, gru_input_dim, kernel_size=1)

        # --- 4. 保留原始 Generator 的 GRU ---
        self.GRU = nn.GRU(gru_input_dim, gru_input_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

        # --- 5. 【新增】时间维度压缩层 (Temporal Compression) ---
        initial_upsample_in_ch = h.ch_init_upsample
        self.gru_to_upsample_proj = weight_norm(Conv1d(1024, initial_upsample_in_ch, kernel_size=1))

        # --- 6. 动态计算目标压缩长度 ---
        self.total_upsampling_factor = 1
        for rate in h.upsample_rates:
            self.total_upsampling_factor *= rate
        self.target_output_length = 130
        ideal_input_length = self.target_output_length / self.total_upsampling_factor
        self.compressed_seq_len = int(math.ceil(ideal_input_length))
        self.temporal_compressor = nn.AdaptiveAvgPool1d(self.compressed_seq_len)

        # --- 7. 保留原始 Generator 的上采样部分 ---
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            in_ch = h.ch_init_upsample // (2 ** i)
            out_ch = h.ch_init_upsample // (2 ** (i + 1))
            self.ups.append(weight_norm(ConvTranspose1d(in_ch, out_ch, k, u, padding=(k - u) // 2)))

        self.conv_mid1 = weight_norm(
            Conv1d(h.ch_init_upsample // (2 ** self.i_mid), h.ch_init_upsample // (2 ** self.i_mid), 3, 1, padding=0))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.ch_init_upsample // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, h.out_ch, 9, 1, padding=get_padding(9, 1)))

        # 初始化
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)
        self.gru_to_upsample_proj.apply(init_weights)
        self.mamba_proj_in.apply(init_weights)  # 初始化新增层

    def forward(self, x):
        # --- 1. 通过 Backbone ---
        x = self.backbone(x)  # x: (B, 32, T/4, C/4)

        # --- 2. 【修复点】Mamba 处理 ---
        # 1. 重塑特征以适应 Mamba (Batch, Sequence_Length, Features)
        B, C, H, W = x.shape
        # 将空间和时间维度展平为序列
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, H*W, 32)

        # 2. 投影到 Mamba 维度
        x = self.mamba_proj_in(x)  # (B, H*W, 8)

        # 3. 通过 Mamba 层 (捕捉长距离依赖)
        x = self.mamba(x)[0]  # (B, H*W, 8)

        # 4. 投影回原始维度
        x = self.mamba_proj_out(x)  # (B, H*W, 32)

        # 5. 恢复为 Conv2d 格式 (为了后续的 backbone_to_gru_proj)
        x = x.view(B, H, W, 32).permute(0, 3, 1, 2).contiguous()  # (B, 32, H, W)

        # --- 3. 重塑 Backbone 特征以适配 GRU (保持不变) ---
        B, C, H, W = x.shape
        new_time_steps = H * W
        x = x.view(B, C, new_time_steps)  # (B, 32, H*W)

        # --- 4. 通过桥接投影层 (改变特征维度) ---
        x = self.backbone_to_gru_proj(x.unsqueeze(-1)).squeeze(-1)  # (B, gru_input_dim, H*W)
        x = x.transpose(1, 2)  # (B, H*W, gru_input_dim)

        # --- 5. 通过原始 GRU ---
        x_temp = x  # 保存一份用于残差连接
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)  # x: (B, H*W, gru_input_dim * 2) 因为双向

        # GRU 输出后，再次转回 Conv1d 格式 (B, channels, time_steps)
        x = x.transpose(1, 2)  # (B, gru_input_dim * 2, H*W)
        x_temp = x_temp.transpose(1, 2)  # (B, gru_input_dim, H*W)
        x = torch.cat([x, x_temp], dim=1)  # (B, gru_input_dim * 3, H*W)

        # --- 6. 【关键修改】压缩时间维度 ---
        x = self.gru_to_upsample_proj(x)  # (B, initial_upsample_in_ch, H*W)
        x = self.temporal_compressor(x)  # (B, initial_upsample_in_ch, compressed_seq_len)

        # --- 7. 通过原始上采样和残差块 ---
        for i in range(self.num_upsamples):
            if i == self.i_mid:
                x = self.conv_mid1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                res_output = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = res_output
                else:
                    xs += res_output
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_mid1)
        remove_weight_norm(self.gru_to_upsample_proj)
        remove_weight_norm(self.mamba_proj_in)  # 移除新增层的 weight norm (如果有的话)


# ==========================================
# Discriminator (保持不变)
# ==========================================
class Discriminator(torch.nn.Module):
    # (此处代码保持不变，为了节省篇幅且你提到问题在 Generator)
    # ... (保留原文档中的 Discriminator 代码) ...
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
        self.conv_pre = weight_norm(Conv1d(h.in_ch, h.ch_init_downsample, 3, 1, padding=get_padding(3, 1)))
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample * (2 ** i), h.ch_init_downsample * (2 ** (i + 1)), k, u,
                       padding=math.ceil((k - u) / 2))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample * (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.GRU = nn.GRU(ch, ch // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9, 1)))

        # FC Layer
        self.adv_classifier = nn.Sequential(nn.Linear(h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), 1),
                                            nn.Sigmoid())
        self.aux_classifier = nn.Sequential(
            nn.Linear(h.ch_init_downsample * 2 * 8 * (self.input_size // self.m), h.n_classes), nn.Softmax(dim=1))

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
        x = x.view(-1, self.ch_init_downsample * 2 * 8 * (self.input_size // self.m))
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