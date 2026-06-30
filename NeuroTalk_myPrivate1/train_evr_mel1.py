import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import argparse
import time
import torchaudio
# import wavio # 仅在生成阶段需要
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wavio
# 引入生成阶段所需的模块
from models.models_HiFi import Generator as model_HiFi
from modules import GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
# from modules import mel2wav_vocoder
from utils import data_denorm, word_index
# from NeuroTalkDataset import myDataset # 不再在阶段1使用
from torchmetrics import CharErrorRate

# 旧代码依赖
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import random

# from kymatio import Scattering1D # 如果旧代码没用到可注释
# from mamba.mamba import Mamba

# ==============================
# 1. 模型定义 (合并版)
# ==============================

# --- 旧代码中的模块 (用于阶段1) ---
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
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout_rate))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y0 = self.conv1(x)
        fused = y0 + x
        fused = self.relu(fused)
        y1 = self.conv2(fused)
        output = y1 + fused
        return self.relu(output)


# Mamba 占位符 (确保环境兼容)
try:
    from mamba.mamba import Mamba

    HAS_MAMBA = True
except ImportError:
    print("⚠️ Warning: mamba-ssm not found. Using dummy block.")
    HAS_MAMBA = False


    class Mamba(nn.Module):
        def __init__(self, **kwargs): super().__init__()

        def forward(self, x): return [x]


# --- 旧代码的主分类模型 (用于阶段1) ---
class EVRNet_Classifier_Old(nn.Module):
    def __init__(self, num_classes):
        super(EVRNet_Classifier_Old, self).__init__()
        self.mamba = Mamba(num_layers=1, d_input=32, d_model=8, d_state=8, d_discr=16, ker_size=4,
                           parallel=False) if HAS_MAMBA else nn.Identity()

        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)
        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)
        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.fc_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.spatial_block1(x)
        x = self.mkrb1(x)
        x = self.temporal_block1(x)
        x = self.mkrb2(x)
        x = self.temporal_block3(x)
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        if HAS_MAMBA:
            x = x.unsqueeze(-2)
            x = self.mamba(x)[0]
            x = x.view(x.size(0), -1)

        x = self.fc_layer(x)
        return x


# --- 新代码的生成器 (用于阶段2) ---
# 注意：这里复用了上面的 SpatialBlock, TemporalBlock, MKRB
class EVRNet_Backbone(nn.Module):
    def __init__(self):
        super(EVRNet_Backbone, self).__init__()
        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)
        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)
        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.spatial_block1(x.transpose(1, 2).unsqueeze(1))  # (B, 32, 96, 24)
        # 这里的逻辑需与新代码完全一致
        # 注意：旧代码输入可能是 (B, 1, 192, 24)，新代码可能有转置
        # 确保这里处理后的维度与新代码一致
        # x = self.spatial_block1(x)
        x = self.mkrb1(x)
        x = self.temporal_block1(x)
        x = self.mkrb2(x)
        x = self.temporal_block3(x)
        return x


class EVRNet_MelGenerator(nn.Module):
    def __init__(self, mel_bins=80):
        super(EVRNet_MelGenerator, self).__init__()
        self.backbone = EVRNet_Backbone()
        # self.time_steps_out = 24  # 示例值
        self.mel_bins = mel_bins
        self.feature_dim = 1504  # 需根据实际输出调整

        self.mel_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, mel_bins)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, C, T, F = features.shape
        features = features.permute(0, 2, 1, 3).contiguous().view(B * T, C * F)
        mel_pred = self.mel_head(features)
        mel_pred = mel_pred.view(B, T, self.mel_bins)
        return mel_pred.permute(0, 2, 1)


# ==============================
# 2. 数据处理 (复用旧代码逻辑)
# ==============================

word_to_label = {
    "my": 8, "dad": 0, "is": 1, "a": 2, "policeman": 3,
    "he": 4, "will": 5, "always": 6, "become": 7, "hero": 9
}


def process_file_fast(file_path, image_data, labels):
    # 简化版数据处理，去除不必要的 I/O 操作
    # 假设文件路径相对于当前目录或在特定数据文件夹下
    # 这里为了演示，假设文件存在。实际运行时请确保路径正确
    try:
        with open(file_path, mode='r') as f:
            lines = f.readlines()
            ls = []
            for line in lines[1282:52844]:
                line_list = line.strip().split('\t')
                columns = [float(line_list[linetime]) for linetime in range(1, 25)]
                ls.append(columns)

            eeg = np.array(ls)

            # 参数
            word_duration = 1.5
            rest_within_word = 5
            rest_between_words = 10
            sampling_rate = 128

            words = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]
            repetitions = 5

            samples_per_word = int(word_duration * sampling_rate)
            samples_per_rest_within_word = int(rest_within_word * sampling_rate)
            samples_per_rest_between_words = int(rest_between_words * sampling_rate)

            word_data_segments = []
            current_index = 0

            for word in words:
                for _ in range(repetitions):
                    word_start = current_index
                    word_end = word_start + samples_per_word
                    word_segment = eeg[word_start:word_end, :]
                    word_data_segments.append(word_segment)

                    label = word_to_label[word]
                    labels.append(label)

                    if _ < repetitions - 1:
                        current_index += samples_per_word + samples_per_rest_within_word
                    else:
                        current_index += samples_per_word + samples_per_rest_between_words

            for eeg_seg in word_data_segments:
                scaler = StandardScaler()
                segment_normalized = scaler.fit_transform(eeg_seg.T).T
                image_data.append(segment_normalized.T)  # 存储为 (C, T)

    except FileNotFoundError:
        print(f"⚠️ Warning: File {file_path} not found. Skipping.")


# ==============================
# 3. 训练流程控制
# ==============================

def run_fast_classification(args, device):
    print("\n" + "=" * 50)
    print("🚀 Phase 1: Fast Classification (Old Logic)")
    print("=" * 50)

    # 1. 数据加载 (旧逻辑)
    image_data = []
    labels = []
    # 使用旧代码的文件列表
    file_paths = ['vDLY-001_1.txt', 'vLCL-001_1.txt', 'vLHJ-001_1.txt', 'vLHJ-002_1.txt',
                  'vLCL-004_1.txt', 'vYHC02_1.txt', 'vLCL-01_1.txt', 'vLCL-02_1.txt',
                  'vCLB-001_1.txt', 'vYHC-001_1.txt', 'vLCL-003_1.txt']

    # 假设数据在 ./orign_eeg_data/ 下，请根据实际情况修改
    data_dir = "./orign_eeg_data/"
    for fp in file_paths:
        process_file_fast(os.path.join(data_dir, fp), image_data, labels)

    if not image_data:
        print("❌ No data loaded. Please check file paths.")
        return None

    image_data_array = np.array(image_data)
    labels_array = np.array(labels)

    # 划分数据集 (旧逻辑)
    test_indices = [i for i in range(0, image_data_array.shape[0], 10)]
    train_data = np.delete(image_data_array, test_indices, axis=0)
    test_data = image_data_array[test_indices]
    train_labels = np.delete(labels_array, test_indices, axis=0)
    test_labels = labels_array[test_indices]

    # 转 Tensor
    train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # (B, 1, T, C)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.batch_size, shuffle=False)

    # 2. 模型初始化
    num_classes = len(word_to_label)
    model = EVRNet_Classifier_Old(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 3. 训练循环 (简化版，去掉了绘图和复杂的指标计算以加快速度)
    best_acc = 0.0
    epochs = 2000  # 旧代码用了2000，这里为了演示快一点设为500，可调回2000

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Cls Ep {epoch + 1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        test_acc = 100 * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            # 【关键点】：保存 backbone 权重供生成器使用
            # 我们需要保存 model 中对应 backbone 的部分
            # 由于类定义不同，我们需要手动映射或修改保存策略
            # 这里我们直接保存整个 state_dict，加载时只加载匹配的部分
            torch.save(model.state_dict(), os.path.join(args.savemodel, 'backbone_pretrained.pth'))
            print(f"✨ New Best Backbone Saved! (Acc: {best_acc:.2f}%)")

    return model


def save_wav_samples(args, model_g, vocoder, model_STT, decoder_STT, val_loader, epoch, device):
    """
    在验证阶段生成音频并保存，用于检查生成效果
    """
    print(f"\n🎧 Generating validation samples for Epoch {epoch}...")
    model_g.eval()
    vocoder.eval()
    model_STT.eval()

    # 获取一个批次的数据
    try:
        # 注意：这里假设 val_loader 返回的数据格式与训练时一致
        # (input, target, target_cl, voice, data_info)
        input, target, target_cl, voice, data_info = next(iter(val_loader))
    except StopIteration:
        print("⚠️ Validation loader is empty.")
        return

    input = input.to(device)
    voice = torch.squeeze(voice, dim=-1).to(device)
    labels = torch.argmax(target_cl, dim=1)

    with torch.no_grad():
        # 1. 生成梅尔谱
        mel_out = model_g(input)

        # 2. 简单的插值对齐 (如果维度不一致)
        if mel_out.shape[-1] != target.shape[-1]:
            mel_out = F.interpolate(mel_out, size=target.shape[-1], mode='linear', align_corners=False)

        # 3. 反归一化
        try:
            output_denorm = data_denorm(mel_out, data_info[0].to(device), data_info[1].to(device))
        except:
            output_denorm = mel_out  # 兜底

        # 4. 声码器生成音频 (HiFi-GAN)
        # 取批次中的第一个样本进行保存
        wav_recon = vocoder(output_denorm[0:1])  # (1, 1, Time)
        if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
            wav_recon = wav_recon.squeeze(1)  # (1, Time)

        # 5. 重采样到 STT 采样率 (16kHz) 以便播放和对比
        # 原始生成是 22050Hz
        wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)

        # 6. 简单的 CTC 解码查看预测文本 (可选)
        # 注意：这里为了简单，只处理第一个样本
        emission_recon, _ = model_STT(wav_recon)
        transcript = decoder_STT(emission_recon[0])

        # 7. 获取真实标签
        gt_label_str = args.word_label[labels[0].item()]

        # 8. 处理文件名
        # 替换 "|" 和空格，防止文件名非法
        str_tar = gt_label_str.replace("|", "").replace(" ", "_")
        str_pred = transcript.replace("|", "").replace(" ", "_").replace(" ", "")  # CTC 输出可能包含空格

        # 限制文件名长度
        title = f"Tar_{str_tar}-Pred_{str_pred}"
        if len(title) > 100:  title = title[:100]

        save_path = os.path.join(args.savevoice, f"e{epoch}_{title}.wav")

        # 9. 保存文件
        # wavio.write 需要 numpy 数组
        wav_data = wav_recon.squeeze(0).cpu().numpy()
        # 确保数据类型正确，通常是 float32 或 int16
        # wavio 会自动处理，但最好归一化到 -1~1 或 0~32767
        # 这里直接使用 wavio 的默认处理
        try:
            wavio.write(save_path, wav_data, args.sample_rate_STT, sampwidth=1)
            print(f"✅ Saved audio sample: {save_path}")
            print(f"   Ground Truth: {gt_label_str} | Predicted: {transcript}")
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")

def run_mel_generation(args, device, writer):
    print("\n" + "=" * 50)
    print("🎵 Phase 2: Mel Generation (New Logic)")
    print("=" * 50)

    # 1. 加载配置
    with open(args.config) as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # 2. 初始化 STT Bundle (必须)
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    args.sample_rate_STT = bundle.sample_rate
    # 确保 word_index 已定义，若未定义需处理
    if not hasattr(args, 'word_index'):
        args.word_index, args.word_length = word_index(args.word_label, bundle)

    # 3. 加载 Vocoder
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    vocoder = model_HiFi(h).to(device)
    vocoder.load_state_dict(torch.load(args.vocoder_pre)['generator'])
    vocoder.eval()
    for p in vocoder.parameters(): p.requires_grad = False

    # 4. 加载 STT
    model_STT = bundle.get_model().to(device)
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    model_STT.eval()
    for p in model_STT.parameters(): p.requires_grad = False

    # 5. 初始化生成器
    model_g = EVRNet_MelGenerator(mel_bins=args.n_mel_channels).to(device)

    # 【关键点】：加载阶段1训练的权重
    bp_path = os.path.join(args.savemodel, 'backbone_pretrained.pth')
    if os.path.exists(bp_path):
        cls_state_dict = torch.load(bp_path)
        # 过滤 backbone 权重
        backbone_dict = {}
        for k, v in cls_state_dict.items():
            if k in model_g.backbone.state_dict():
                backbone_dict[k] = v

        model_g.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"✅ Loaded backbone weights from Phase 1.")
    else:
        print("⚠️ No Phase 1 weights found. Training from scratch.")

    # 6. 数据加载
    try:
        from NeuroTalkDataset import myDataset
        trainset = myDataset(mode=0, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
        valset = myDataset(mode=2, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    except ImportError:
        print("❌ NeuroTalkDataset not found. Please ensure it's in path for Phase 2.")
        return

    # 7. 训练设置
    optimizer_g = optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay)
    criterion_recon = RMSELoss().to(device)
    criterion_ctc = nn.CTCLoss().to(device)

    best_loss = 1e9

    # ==========================
    # 训练循环 (修正了缩进)
    # ==========================
    for epoch in range(args.max_epochs):
        model_g.train()
        vocoder.eval()
        model_STT.eval()

        epoch_loss = []
        pbar = tqdm(train_loader, desc=f"Gen Ep {epoch + 1}/{args.max_epochs}")

        for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)
            voice = torch.squeeze(voice, dim=-1).to(device)
            labels = torch.argmax(target_cl, dim=1)

            # --- 修正部分开始 ---
            # 1. 准备 CTC 标签 (在循环外收集整个批次)
            gt_label_idx = []
            gt_length = []
            for j in range(len(labels)):
                gt_label_idx.append(args.word_index[labels[j].item()])
                gt_length.append(args.word_length[labels[j].item()])

            # 转换为 Tensor 并移到设备
            gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64).to(device)
            gt_length = torch.tensor(gt_length, dtype=torch.int64).to(device)
            # --- 修正部分结束 ---

            optimizer_g.zero_grad()

            # 2. Generate Mel
            mel_out = model_g(input)  # (B, 80, T_out)

            # 3. Align (简单插值对齐)
            if mel_out.shape[-1] != target.shape[-1]:
                mel_out = F.interpolate(mel_out, size=target.shape[-1], mode='linear', align_corners=False)

            # 4. Reconstruction Loss
            loss_recon = criterion_recon(mel_out, target)

            # 5. CTC Loss (Vocoder -> STT)
            with torch.no_grad():
                # 简单的反归一化
                try:
                    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
                except:
                    output_denorm = mel_out  # 兜底

            # Vocoder
            try:
                wav_recon = vocoder(output_denorm)
                if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
                    wav_recon = wav_recon.squeeze(1)
            except Exception as e:
                print(f"Vocoder Error: {e}")
                continue

            # Resample for STT
            if wav_recon.shape[1] != voice.shape[1]:
                wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)
                if wav_recon.shape[1] < voice.shape[1]:
                    pad = voice.shape[1] - wav_recon.shape[1]
                    wav_recon = F.pad(wav_recon, (0, pad))
                elif wav_recon.shape[1] > voice.shape[1]:
                    wav_recon = wav_recon[:, :voice.shape[1]]

            # STT Forward
            emission_recon, _ = model_STT(wav_recon)
            emission_recon_ = emission_recon.log_softmax(2)

            # CTC Loss
            input_lengths = torch.full(size=(emission_recon.size(0),), fill_value=emission_recon.size(1),
                                       dtype=torch.long).to(device)
            loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)

            # Total Loss
            loss_g = args.l_g[0] * loss_recon + args.l_g[2] * loss_ctc
            loss_g.backward()

            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            optimizer_g.step()

            epoch_loss.append(loss_g.item())
            pbar.set_postfix({'L_tot': f'{np.mean(epoch_loss):.4f}'})

        # Epoch End (在批次循环结束后)
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # # Save Checkpoint
        # if (epoch + 1) % args.val_interval == 0:
        #     is_best = avg_loss < best_loss
        #     if is_best:
        #         best_loss = avg_loss
        #         state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
        #         save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        #         print(f"✨ Best Gen Model Saved! Loss: {best_loss:.4f}")

        # ==============================
        # 【新增】验证与保存音频
        # ==============================
        # 每隔 val_interval 个 epoch 保存一次音频，或者每个 epoch 都保存
        if (epoch + 1) % args.val_interval == 0:
            # 1. 保存检查点 (原有逻辑)
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
                save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
                print(f"✨ Best Gen Model Saved! Loss: {best_loss:.4f}")

            # 2. 保存音频文件 (新逻辑)
            # 传入 val_loader 用于获取一个批次的数据
            save_wav_samples(args, model_g, vocoder, model_STT, decoder_STT, val_loader, epoch, device)

        scheduler_g.step()

    print("🎉 Phase 2 Complete!")

# ==============================
# 4. 主程序入口
# ==============================


def main(args):
        device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')

        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 设置GPU的随机种子
        np.random.seed(seed)  # NumPy的随机种子
        random.seed(seed)  # Python内置的随机模块种子
        torch.backends.cudnn.deterministic = True  # 确保每次返回相同的结果

        # 目录设置
        args.logDir = os.path.abspath(args.logDir)
        saveDir = os.path.join(args.logDir, f"{args.sub}_{args.task}_EVR_Hybrid")
        args.savemodel = os.path.join(saveDir, 'savemodel')
        args.savevoice = os.path.join(saveDir, 'epovoice')
        args.logs = os.path.join(saveDir, 'logs')

        os.makedirs(args.savemodel, exist_ok=True)
        os.makedirs(args.savevoice, exist_ok=True)
        os.makedirs(args.logs, exist_ok=True)

        writer = SummaryWriter(args.logs)

        # ==========================
        # PHASE 1: 快速分类训练 (旧逻辑)
        # ==========================
        # 注意：这里我们直接调用旧逻辑训练，不依赖复杂的 Dataset 类
        print("\n🚀 Starting Phase 1 (Fast Classification)...")
        cls_model = run_fast_classification(args, device)

        if cls_model is None:
            print("❌ Phase 1 Failed or Skipped. Exiting.")
            return

        # 清理显存
        del cls_model
        torch.cuda.empty_cache()

        # ==========================
        # PHASE 2: 语音生成训练 (新逻辑)
        # ==========================
        print("\n🎵 Starting Phase 2 (Mel Generation)...")
        run_mel_generation(args, device, writer)

        writer.close()
        print("🏁 All Training Complete!")

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Hybrid EVRNet Training')
        # 基础参数
        parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000')
        parser.add_argument('--dataLoc', type=str, default='./processed_dataset_1')  # 生成阶段数据
        parser.add_argument('--config', type=str, default='./config_myPrivate.json')
        parser.add_argument('--logDir', type=str, default='./TrainResult_EVR')
        parser.add_argument('--gpuNum', type=list, default=[0])
        parser.add_argument('--batch_size', type=int, default=24)
        parser.add_argument('--sub', type=str, default='sub1')
        parser.add_argument('--task', type=str, default='SpokenEEG')
        parser.add_argument('--recon', type=str, default='Voice_mel')

        # 训练参数
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--max_epochs', type=int, default=1000)  # 生成阶段轮数
        parser.add_argument('--val_interval', type=int, default=5)

        # 模型参数 (需与config一致或在这里覆盖)
        parser.add_argument('--n_mel_channels', type=int, default=80)
        parser.add_argument('--sample_rate_mel', type=int, default=22050)
        parser.add_argument('--lr_g', type=float, default=1e-4)
        parser.add_argument('--lr_g_decay', type=float, default=0.995)
        parser.add_argument('--l_g', type=list, default=[1, 0.01, 0.01])  # 权重: [recon, ..., ctc]

        # 标签 (需与旧代码一致)
        parser.add_argument('--word_label', type=list,
                            default=["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "hero"])

        args = parser.parse_args()
        main(args)