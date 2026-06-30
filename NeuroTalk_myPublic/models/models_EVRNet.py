import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from utils import init_weights, get_padding
import math

LRELU_SLOPE = 0.1

# 定义各个模块
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__()

        # First part (3x3 conv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Second part (5x5 conv)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # 第一个卷积层提取特征向量 y0
        y0 = self.conv1(x)

        # 将 y0 与输入 x 融合
        fused = y0 + x

        fused = self.relu(fused)

        # 第二个卷积层进一步提取特征
        y1 = self.conv2(fused)

        # 输出最终的特征向量
        output = y1 + fused  # 残差连接

        return self.relu(output)

from mamba.mamba import Mamba


class EVRNet_Generator(nn.Module):
    def __init__(self, h=None, n_mels=80, output_time_frames=130):
        """
        初始化 EVRNet Generator.

        参数:
            h: Config 对象 (AttrDict)。如果提供，将优先从中读取 out_ch 等参数。
            n_mels: 目标 Mel 通道数 (默认 80)。
            output_time_frames: 目标时间帧数 (默认 129，对应 1.5s @ 22050Hz / 256hop).
        """
        super(EVRNet_Generator, self).__init__()

        # 1. 参数解析 (兼容 Config 对象或直接传参)
        if h is not None:
            self.target_mels = int(h.out_ch) if hasattr(h, 'out_ch') else n_mels
            # 尝试从 config 推断时间长度，如果没有则使用默认值
            # 注意：config_g.json 通常没有直接的 output_time_frames，可能需要根据 input_size 或硬编码
            if hasattr(h, 'input_size'):
                # 如果 discriminator 的 input_size 已知，通常 generator 输出要匹配它
                self.target_frames = int(h.input_size)
            else:
                self.target_frames = output_time_frames
        else:
            self.target_mels = n_mels
            self.target_frames = output_time_frames

        print(f"[EVRNet Init] Target: {self.target_mels} mels, {self.target_frames} frames")

        # 2. 空间特征提取 (EEG 通道间关系)
        # 输入: (B, 1, 192, 24) -> 这里的 1 是伪通道，192是时间，24是电极
        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)

        # 3. 时间特征提取 (EEG 时序关系)
        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)
        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)


        # 修正方案：使用 AdaptiveAvgPool2d 将最后一个维度 (电极/空间残留) 压为 1
        self.avg_pool_spatial = nn.AdaptiveAvgPool2d((None, 1))

        # Mamba 定义
        d_input_mamba = 32  # 对应卷积输出的通道数
        d_model_mamba = 32

        self.mamba = Mamba(
            num_layers=3,
            d_input=d_input_mamba,
            d_model=d_model_mamba,
            d_state=16,
            d_discr=16,
            ker_size=4,
            parallel=False,
        )

        # 5. 映射到 Mel 频谱
        # 不再使用 ConvTranspose1d 进行粗糙上采样，改用 Interpolate 精确控制长度
        self.mel_projection = nn.Sequential(
            nn.Conv1d(in_channels=d_input_mamba, out_channels=self.target_mels, kernel_size=3, padding=1),
        )

    def forward(self, x):

        """
        x: (Batch, 1, Time_In, Channels_In)
           例如: (B, 1, 192, 24)
        """
        # --- 1. 特征提取 ---
        # 注意：请确保你的 SpatialBlock 和 TemporalBlock 内部逻辑与这里的维度变化一致
        # 如果你的块定义是 Conv2d(kernel=(k, 1))，它是在第 2 维 (Time) 滑动。
        # 如果你的块定义是 Conv2d(kernel=(1, k))，它是在第 3 维 (Channel) 滑动。

        # 假设流程如下 (基于你提供的注释逻辑):
        x = self.spatial_block1(x.transpose(1, 2).unsqueeze(1))  # (B, 32, 96, 24)  [假设下采样了 Time]
        x = self.mkrb1(x)  # (B, 32, 96, 24)

        x = self.temporal_block1(x)  # (B, 32, 96, 12)  [假设下采样了 Channel/Electrode]
        x = self.mkrb2(x)  # (B, 32, 96, 12)

        x = self.temporal_block3(x)  # (B, 32, 48, 6)   [假设再次下采样了 Time]

        # 此时 x 形状: (B, 32, 48, 6)
        # 时间维度 (Seq Len) 是 48。空间/电极维度是 6。

        # --- 2. 准备 Mamba 输入 ---
        # 目标: (Batch, Seq_Len, Input_Dim)
        # 当前: (B, 32, 48, 6) -> 我们希望 Seq_Len=48, Input_Dim=32*6? 或者 Pool 掉 6?

        # 方案 A: Pool 掉最后的维度 (6 -> 1)，保留通道 32 作为特征维
        x = self.avg_pool_spatial(x)  # (B, 32, 48, 1)
        x = x.squeeze(-1)  # (B, 32, 48) -> (Batch, Channels, Time)

        # Mamba 需要 (Batch, Time, Channels)
        x = x.permute(0, 2, 1)  # (B, 48, 32)

        # --- 3. Mamba 处理 ---
        # mamba 返回 (output, state)
        x, _ = self.mamba(x)  # (B, 48, 64)

        # --- 4. 上采样与投影 ---
        # 转回 (Batch, Channels, Time) 以便 Conv1d/Interpolate
        x = x.permute(0, 2, 1)  # (B, 64, 48)

        # 【核心修复】使用 interpolate 精确调整时间维度到 target_frames (129)
        # mode='linear' 对于 1D 序列是最自然的上采样方式
        # print(x.shape)
        x = F.interpolate(x, size=self.target_frames, mode='linear', align_corners=False)
        # 现在 x 形状: (B, 64, 129)
        # print(x.shape)

        # 投影到 Mel 通道
        x = self.mel_projection(x)  # (B, 80, 129)
        # print(x.shape)
        # print('=========================================')

        return x

class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1,3,5)):
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
            
            
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_upsample//2,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        
        self.GRU = nn.GRU(h.ch_init_upsample//2, 
                          h.ch_init_upsample//4, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, 
                                       h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.ch_init_upsample//(2**i), 
                                h.ch_init_upsample//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
            
        self.conv_mid1 = weight_norm(
            Conv1d(h.ch_init_upsample//(2**self.i_mid), 
                   h.ch_init_upsample//(2**self.i_mid), 
                   3, 1, 
                   padding=0))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.ch_init_upsample//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(ch, 
                   h.out_ch, 
                   9, 1, 
                   padding=get_padding(9,1)))
        
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        for i in range(self.num_upsamples):
            # to match the output size
            if i == self.i_mid:
                x = self.conv_mid1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
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
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_mid1)


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
                   padding=get_padding(3,1)))
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, 
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample*(2**i), 
                       h.ch_init_downsample*(2**(i+1)),
                       k, u, padding=math.ceil((k-u)/2))))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample*(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        
        self.GRU = nn.GRU(ch, ch//2,
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9,1)))
        
        # FC Layer 
        self.adv_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), 1),
            nn.Sigmoid())
        self.aux_classifier = nn.Sequential(nn.Linear(
            h.ch_init_downsample*2*8*(self.input_size//self.m), h.n_classes),
            nn.Softmax(dim=1))

        # 【新增】计算模型期望的下采样后时间长度
        # 原公式: dim = ch * 2 * 8 * (time_len) = 6144
        # 我们反推 time_len
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

        # ================= 修改开始 =================
        # 【核心修复】使用自适应平均池化，强制将时间维度 (最后一维)
        # 压缩到模型初始化时预期的长度 (self.expected_time_len)
        # 这样无论输入音频多长，展平后的特征数都会固定，避免 RuntimeError
        x = F.adaptive_avg_pool1d(x, self.expected_time_len)
        # ================= 修改结束 =================

        # FC Layer
        # 现在这里可以安全运行了，因为维度已经对齐
        x = x.view(-1,
                   self.ch_init_downsample
                   * 2 * 8 * (self.input_size // self.m))
        validity = self.adv_classifier(x)
        label = self.aux_classifier(x)

        return validity, label

    # def forward(self, x):
    #     x = self.conv_pre(x)
    #
    #     for i in range(self.num_downsamples):
    #         x = F.leaky_relu(x, LRELU_SLOPE)
    #         x = self.downs[i](x)
    #
    #         xs = None
    #         for j in range(self.num_kernels):
    #             if xs is None:
    #                 xs = self.resblocks[i*self.num_kernels+j](x)
    #             else:
    #                 xs += self.resblocks[i*self.num_kernels+j](x)
    #         x = xs / self.num_kernels
    #     x = F.leaky_relu(x)
    #     x_temp = x
    #     x = x.transpose(1, 2)
    #     self.GRU.flatten_parameters()
    #     x, _ = self.GRU(x)
    #     x = x.transpose(1, 2)
    #     x = torch.cat([x, x_temp], dim=1)
    #
    #     # FC Layer
    #     x = x.view(-1,
    #                self.ch_init_downsample
    #                *2*8*(self.input_size//self.m))
    #     validity = self.adv_classifier(x)
    #     label = self.aux_classifier(x)
    #
    #     return validity, label

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
            