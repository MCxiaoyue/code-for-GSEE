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
import wavio
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from modules import mel2wav_vocoder, perform_STT
from torch.optim.lr_scheduler import MultiStepLR # 添加这一行
import mne
from sklearn.metrics import f1_score # 请确保在文件开头或此处导入
# ==============================
# 引入必要的模块
# ==============================

# 引入旧代码中的复杂模型 (确保 models/models.py 存在)
try:
    from models.models1_1_lowDis_4_public import Discriminator, EVRNet_Generator
    from models.models_HiFi import Generator as model_HiFi
except ImportError:
    print("❌ Error: Cannot import models. Please ensure 'models/models.py' exists.")
    exit()

# 引入工具模块
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from utils import data_denorm, word_index, init_weights, get_padding
from torchmetrics import CharErrorRate

# 旧代码依赖 (Phase 1)
from sklearn.preprocessing import StandardScaler

# Mamba 支持 (Phase 1)
try:
    from mamba.mamba import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


    class Mamba(nn.Module):
        def __init__(self, **kwargs): super().__init__()

        def forward(self, x): return [x]


# ==============================
# 1. Phase 1: 分类模型定义 (保持不变)
# ==============================

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__( )

        # First part (3x3 conv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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


class EVRNet_Classifier_Old(nn.Module):
    def __init__(self, num_classes):
        super(EVRNet_Classifier_Old, self).__init__()
        self.mamba = Mamba(
            num_layers=1,  # Number of layers of the full model
            d_input=64,  # Dimension of each vector in the input sequence (i.e. token size)
            d_model=24,  # Dimension of the visible state space
            d_state=8,  # Dimension of the latent hidden states
            d_discr=16,  # Rank of the discretization matrix Δ
            ker_size=4,  # Kernel size of the convolution in the MambaBlock
            parallel=False,  # Whether to use the sequenial or the parallel implementation
        )

        # self.mamba = Mamba(
        #     num_layers=1,  # Number of layers of the full model
        #     d_input=64,   # Dimension of each vector in the input sequence (i.e. token size)
        #     d_model=32,  # Dimension of the visible state space
        #     d_state=32,  # Dimension of the latent hidden states
        #     d_discr=32,  # Rank of the discretization matrix Δ
        #     ker_size=4,  # Kernel size of the convolution in the MambaBlock
        #     parallel=False,  # Whether to use the sequenial or the parallel implementation
        # )
        self.temporal_block1 = TemporalBlock(1, 32, kernel_size=4, stride=3)
        # self.dropout1 = nn.Dropout2d(dropout_prob)  # 在第一个TemporalBlock后添加Dropout

        self.mkrb1 = MKRB(32, 32)
        # self.dropout2 = nn.Dropout2d(dropout_prob)  # 在MKRB1后添加Dropout

        self.spatial_block1 = SpatialBlock(32, 32, kernel_size=4, stride=3)
        # self.dropout3 = nn.Dropout2d(dropout_prob)  # 在第一个SpatialBlock后添加Dropout

        self.mkrb2 = MKRB(32, 32)
        # self.dropout4 = nn.Dropout2d(dropout_prob)  # 在MKRB2后添加Dropout

        self.spatial_block2 = SpatialBlock(32, 64, kernel_size=4, stride=3)
        # self.dropout5 = nn.Dropout2d(dropout_prob)  # 在第二个SpatialBlock后添加Dropout

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层
        self.fc_layer = nn.Linear(64, num_classes)  # 调整线性层的输入大小
        self.dropout6 = nn.Dropout(0.1)  # 在全连接层前添加Dropout

    def forward(self, x):
        x = self.temporal_block1(x)
        # x = self.dropout1(x)  # 应用Dropout

        x = self.mkrb1(x)
        # x = self.dropout2(x)  # 应用Dropout

        x = self.spatial_block1(x)
        # x = self.dropout3(x)  # 应用Dropout

        x = self.mkrb2(x)
        # x = self.dropout4(x)  # 应用Dropout

        x = self.spatial_block2(x)
        # x = self.dropout5(x)  # 应用Dropout

        x = self.avg_pooling(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.dropout6(x)  # 应用Dropout

        # print(x.shape)

        x = x.unsqueeze(-2)

        # print(x.shape)

        x = self.mamba(x)[0]

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer(x)
        return x


# ==============================
# 2. Phase 1: 数据处理与训练
# ==============================

word_to_label = {
    'flower': 0,
    'penguin': 1,
    'guitar': 2
}

# 定义任务、主题、会话以及路
task = 'audio'
tag = 's'
duration = 2  # 语音刺的持续时间
subjects = ['19']  # 添加所有需要处理的主题ID  '12', '20'   '14', '21'   '15', '22'
sessions = ['1']  # 如果有多个会话的话

def process_file_fast(perception_path, all_image_data, all_labels):
    # 遍历每个主题和会话
    for subject in subjects:
        for session in sessions:
            datapoint = f'{subject}_{session}_epo.fif'

            # perception_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_{task}\\'
            # imagine_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\imagine_{task}\\'

            try:
                # 读取并裁剪感知数据
                perception_epochs = mne.read_epochs(perception_path + datapoint)
                perception_epochs.crop(tmin=0, tmax=duration)

                # # 读取想象数据（如果需要）
                # imagination_epochs = mne.read_epochs(imagine_path + datapoint)
                # imagination_epochs.crop(tmin=0, tmax=duration)

                # 合并感知和想象数据（如果需要）
                epochs = mne.concatenate_epochs([perception_epochs])  # , imagination_epochs

                # 更新事件ID
                event_id_mapping = {
                    'flower': 0,
                    'penguin': 1,
                    'guitar': 2
                }
                for old_event_id, new_event_id in event_id_mapping.items():
                    perc_event_ids = [f'perc_{old_event_id}_{tag}']
                    # imag_event_ids = [f'imag_{old_event_id}_{tag}']
                    epochs = mne.epochs.combine_event_ids(
                        epochs,
                        old_event_ids=perc_event_ids,  # + imag_event_ids
                        new_event_id={old_event_id: new_event_id}
                    )

                # 获取标签
                labels = epochs.events[:, -1]

                # 标准化数据
                image_data_array = np.array(epochs.get_data())
                for i in range(image_data_array.shape[0]):
                    data_to_scale = image_data_array[i, :, :]
                    scaler = StandardScaler()  # MinMaxScaler, RobustScaler, StandardScaler, Normalizer
                    normalized_sample = scaler.fit_transform(data_to_scale)
                    all_image_data.append(normalized_sample)

                    # all_image_data.append(data_to_scale)

                all_labels.extend(labels)

            except FileNotFoundError:
                print(f"File not found for subject {subject}, session {session}. Skipping.")


def run_fast_classification(args, device):
    print("\n" + "=" * 50)
    print("🚀 Phase 1: Fast Classification (Old Logic)")
    print("=" * 50)

    # --- 数据加载部分 (保持不变) ---
    image_data = []
    labels = []
    # 注意：这里 task 变量似乎未定义，可能需要作为参数传入或在函数内定义
    # 假设 task 已经在外部定义，或者从 args 中获取
    perception_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_{task}\\'
    process_file_fast(perception_path, image_data, labels)

    image_data_array = np.array(image_data)
    labels_array = np.array(labels)

    test_indices = [i for i in range(0, image_data_array.shape[0], 10)]
    train_data = np.delete(image_data_array, test_indices, axis=0)
    test_data = image_data_array[test_indices]
    train_labels = np.delete(labels_array, test_indices, axis=0)
    test_labels = labels_array[test_indices]

    train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.batch_size, shuffle=False)

    # --- 模型初始化 (保持不变) ---
    num_classes = len(word_to_label)  # 假设 word_to_label 已定义
    model = EVRNet_Classifier_Old(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_acc = 0.0
    epochs = 250

    # ============ 【新增】定义 Loss 和 F1 记录列表 ============
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # --- 训练集指标记录列表 ---
        all_train_targets = []
        all_train_predictions = []

        # --- 测试集指标记录列表 ---
        all_test_targets = []
        all_test_predictions = []

        # --- 训练部分 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Cls Ep {epoch + 1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            # ============ 【新增】收集训练集 Batch 结果 ============
            all_train_targets.extend(targets.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())

            # 更新进度条显示 (仅显示实时 Loss 和 Acc)
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{100 * correct_train / total_train:.2f}%'})

        # ============ 【新增】计算训练集 F1-Score ============
        train_f1 = f1_score(all_train_targets, all_train_predictions, average='weighted')

        # 计算平均 Loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # --- 测试部分 ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)  # 累计总 Loss

                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

                # ============ 【新增】收集测试集 Batch 结果 ============
                all_test_targets.extend(targets.cpu().numpy())
                all_test_predictions.extend(predicted.cpu().numpy())

        # ============ 【新增】计算测试集 F1-Score ============
        test_f1 = f1_score(all_test_targets, all_test_predictions, average='weighted')

        # 计算测试集平均 Loss 和 准确率
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        test_acc = 100 * test_correct / test_total

        # --- 模型保存与打印 ---
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.savemodel, 'backbone_pretrained.pth'))
            print(f"✨ New Best Backbone Saved! (Acc: {best_acc:.2f}%)")

        # ============ 【修改】打印结果 (增加了 F1-Score) ============
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Test Loss: {test_loss:.4f} "
              f"Train Acc: {100 * correct_train / total_train:.2f}% "
              f"Test Acc: {test_acc:.2f}% "
              f"Train F1: {train_f1:.4f} "
              f"Test F1: {test_f1:.4f}")

    return model


# ==============================
# 3. Phase 2: 生成模型训练 (使用旧代码的复杂结构)
# ==============================
def saveData(args, val_loader, models, epoch):
    """保存生成的音频样本 (包括生成的音频和真实音频) - 修复版 (随机索引)"""
    model_g, _, vocoder, model_STT, decoder_STT = models
    model_g.eval()
    vocoder.eval()
    model_STT.eval()

    try:
        # ==========================
        # 【修改点 1】获取 Batch 数据 (不再在这里取 next)
        # ==========================
        batch_data = next(iter(val_loader))
        # 将 batch 数据解包，但暂时不切片
        input, target, target_cl, voice, data_info = batch_data

    except StopIteration:
        return

    # ==========================================
    # 【新增】随机选取索引
    # ==========================================
    # 获取当前 Batch 的大小 (样本数量)
    batch_size = input.size(0)
    # 随机生成一个索引
    random_idx = random.randint(0, batch_size - 1)
    print(f"📸 Saving random sample (Index: {random_idx}) from batch (Epoch {epoch})")

    # ==========================================
    # 【修改点 2】应用随机索引
    # ==========================================
    # 注意：这里只对 input, target, target_cl, voice 进行切片
    # data_info 通常是标量或列表，不需要切片，保持原样
    input_slice = input[random_idx:random_idx + 1]  # 保持 Batch 维度 (1, ...)
    target_slice = target[random_idx:random_idx + 1]
    target_cl_slice = target_cl[random_idx:random_idx + 1]
    voice_slice = voice[random_idx:random_idx + 1]
    labels_slice = torch.argmax(target_cl_slice, dim=1)

    input_slice = input_slice.to(args.device)
    target_slice = target_slice.to(args.device)
    voice_slice = torch.squeeze(voice_slice, dim=-1).to(args.device)

    with torch.no_grad():
        # ==========================
        # 1. 处理生成样本 (Prediction)
        # ==========================
        output = model_g(input_slice)

        # 确保 target 和 output 在同一设备上
        if output.is_cuda:
            target_slice = target_slice.to(output.device)

        # DTW 对齐
        mel_out = DTW_align(output, target_slice)
        output_denorm = data_denorm(mel_out, data_info[0].to(args.device), data_info[1].to(args.device))

        wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0], dim=0), vocoder, 1)
        wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))
        wav_recon = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)

        if wav_recon.shape[1] != voice_slice.shape[1]:
            p = voice_slice.shape[1] - wav_recon.shape[1]
            p_s = p // 2
            p_e = p - p_s
            wav_recon = F.pad(wav_recon, (p_s, p_e))

        ##### STT Wav2Vec 2.0
        gt_label = args.word_label[labels_slice[0].item()]  # 使用切片后的 label

        transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)

        # save
        wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())

        str_tar = args.word_label[labels_slice[0].item()].replace("|", ",")  # 使用切片后的 label
        str_tar = str_tar.replace(" ", ",")

        str_pred = transcript_recon[0].replace("|", ",")
        str_pred = str_pred.replace(" ", ",")

        title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
        wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT,
                    sampwidth=1)

        # ==========================================
        # 【修改点 3】真实样本 (Ground Truth) 也使用随机索引
        # ==========================================
        # 注意：这里 target 也要换成 target_slice
        target_denorm = data_denorm(target_slice, data_info[0], data_info[1])
        gt_label_list = []
        # 这里循环只跑一次，因为 target_slice 只有一个样本
        for k in range(len(target_slice)):
            gt_label_list.append(args.word_label[labels_slice[k].item()])
        wav_target = mel2wav_vocoder(target_denorm, vocoder, 1)
        wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))
        wav_target = torchaudio.functional.resample(wav_target, args.sampling_rate, args.sample_rate_STT)
        if wav_target.shape[1] != voice_slice.shape[1]:
            p = voice_slice.shape[1] - wav_target.shape[1]
            p_s = p // 2
            p_e = p - p_s
            wav_target = F.pad(wav_target, (p_s, p_e))
        wav_target = wav_target.cpu().detach().numpy()
        title = "Tar_{}".format(str_tar)
        wavio.write(args.savevoice + "/" + title + ".wav", wav_target[0], args.sample_rate_STT, sampwidth=1)


def run_mel_generation(args, device, writer):
    print("\n" + "=" * 50)
    print("🎵 Phase 2: Mel Generation (Complex GAN Logic)")
    print("=" * 50)

    # 1. 加载配置
    with open(args.config) as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    # 设置设备属性以便后续使用
    args.device = device

    # 2. 初始化 STT Bundle
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    args.sample_rate_STT = bundle.sample_rate
    if not hasattr(args, 'word_index'):
        args.word_index, args.word_length = word_index(args.word_label, bundle)

    # 3. 加载 Vocoder
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    vocoder = model_HiFi(h).to(device)
    vocoder.load_state_dict(torch.load(args.vocoder_pre)['generator'])
    vocoder.eval()
    for p in vocoder.parameters():
        p.requires_grad = False

    # 4. 加载 STT
    model_STT = bundle.get_model().to(device)
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    model_STT.eval()
    for p in model_STT.parameters():
        p.requires_grad = False

    # 5. 初始化生成器和判别器 (使用旧代码的复杂结构)
    try:
        config_file_g = os.path.join(args.model_config, 'config_g.json')
        with open(config_file_g) as f:
            h_g = AttrDict(json.load(f))
        if not hasattr(h_g, 'upsample_initial_channel'):
            h_g.upsample_initial_channel = 512

        model_g = EVRNet_Generator(h_g).to(device)
        config_file_d = os.path.join(args.model_config, 'config_d_lowDis+dropout+noise+ch_init_downsample=24.json')
        with open(config_file_d) as f:
            h_d = AttrDict(json.load(f))
        model_d = Discriminator(h_d).to(device)
        print("✅ Loaded Complex Generator & Discriminator from config.")
    except Exception as e:
        print(f"❌ Error loading Generator/Discriminator config: {e}")
        print("⚠️ Please ensure 'models/config_g.json' and 'models/config_d_lowDis+dropout+noise+ch_init_downsample=24.json' exist.")
        return

    # 6. 加载 Phase 1 权重到 Generator
    bp_path = os.path.join(args.savemodel, 'backbone_pretrained.pth')
    if os.path.exists(bp_path):
        cls_state_dict = torch.load(bp_path, map_location=device)
        model_g_dict = model_g.state_dict()
        pretrained_dict = {k: v for k, v in cls_state_dict.items() if k != 'fc_layer.weight' and k != 'fc_layer.bias'}
        model_g_dict.update(pretrained_dict)
        model_g.load_state_dict(model_g_dict, strict=False)
        print(bp_path)
        print("✅ 完美加载！生成器成功继承了分类器的 Backbone + Pool + Mamba 权重")

    # 7. 数据加载
    try:
        from NeuroTalkDataset import myDataset
        trainset = myDataset(mode=0, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
        valset = myDataset(mode=2, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    except ImportError:
        print("❌ NeuroTalkDataset not found.")
        return

    # 8. 优化器与损失函数
    # *** 关键修改：降低判别器学习率 ***
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d * 0.1, betas=(0.8, 0.99),
                                    weight_decay=0.01)  # 降低10倍

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_g = MultiStepLR(optimizer_g, milestones=[150, 250], gamma=0.4)
    # scheduler_g = MultiStepLR(optimizer_g, milestones=[150, 250], gamma=0.2)
    # scheduler_g = MultiStepLR(optimizer_g, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)

    criterion_recon = RMSELoss().to(device)
    criterion_adv = nn.BCELoss().to(device)
    criterion_ctc = nn.CTCLoss().to(device)
    criterion_cl = nn.CrossEntropyLoss().to(device)
    CER = CharErrorRate().to(device)

    best_loss = 1000

    # ==========================
    # 训练循环
    # ==========================

    # *** 新增：设置 G:D 的训练比例 ***
    k_train_g_per_d = 2  # 每训练 1 次 D，就训练 k_train_g_per_d 次 G

    for epoch in range(args.max_epochs):
        model_g.train()
        model_d.train()
        vocoder.eval()
        model_STT.eval()

        # 初始化所有需要记录的详细损失列表 (用于每个epoch的平均计算)
        epoch_loss_g_total = []
        epoch_loss_g_recon = []
        epoch_loss_g_adv = []
        epoch_loss_g_ctc = []
        epoch_loss_d_total = []
        epoch_loss_d_cl = []
        epoch_loss_d_real = []
        epoch_loss_d_fake = []
        epoch_loss_d_real_total = []  # L_D(real, fake)
        epoch_acc_g_adv = []
        epoch_acc_d_real = []
        epoch_acc_d_fake = []

        pbar = tqdm(train_loader, desc=f"Gen Ep {epoch + 1}/{args.max_epochs}")

        for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)
            target_cl = target_cl.to(device)
            voice = torch.squeeze(voice, dim=-1).to(device)
            labels = torch.argmax(target_cl, dim=1)

            # 准备 CTC 标签
            gt_label_idx = []
            gt_length = []
            for j in range(len(labels)):
                gt_label_idx.append(args.word_index[labels[j].item()])
                gt_length.append(args.word_length[labels[j].item()])
            gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64).to(device)
            gt_length = torch.tensor(gt_length, dtype=torch.int64).to(device)

            # --- 训练生成器 k 次 ---
            for _ in range(k_train_g_per_d):
                for p in model_g.parameters():
                    p.requires_grad_(True)
                for p in model_d.parameters():
                    p.requires_grad_(False)  # 冻结 D

                optimizer_g.zero_grad()

                output = model_g(input)
                # print(output.shape)
                # print(target.shape)
                # print('===============================')
                mel_out = DTW_align(output, target)

                # 判别器输出 (用于 GAN Loss)
                g_valid, _ = model_d(mel_out)
                valid = torch.ones((len(input), 1), dtype=torch.float32).to(device)

                # 1. 重建损失 (RMSE)
                loss_recon = criterion_recon(mel_out, target)

                # 2. GAN 损失 (Generator perspective)
                loss_adv = criterion_adv(g_valid, valid)
                acc_g_adv = (g_valid.round() == valid).float().mean()

                # 3. CTC 损失 (通过 Vocoder -> STT)
                with torch.no_grad():
                    output_denorm = data_denorm(mel_out, data_info[0].to(device), data_info[1].to(device))
                    wav_recon = vocoder(output_denorm)
                    if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
                        wav_recon = wav_recon.squeeze(1)
                    voice_for_stt = torchaudio.functional.resample(voice, args.sampling_rate, args.sample_rate_STT)
                    wav_recon_stt = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)

                    max_len = max(voice_for_stt.shape[1], wav_recon_stt.shape[1])
                    voice_for_stt = F.pad(voice_for_stt, (0, max_len - voice_for_stt.shape[1]))
                    wav_recon_stt = F.pad(wav_recon_stt, (0, max_len - wav_recon_stt.shape[1]))

                    emission_recon, _ = model_STT(wav_recon_stt)
                    emission_recon_ = emission_recon.log_softmax(2)
                    input_lengths = torch.full(size=(emission_recon.size(0),), fill_value=emission_recon.size(1),
                                               dtype=torch.long).to(device)

                    loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)

                # 生成器总损失
                loss_g_total = args.l_g[0] * loss_recon + args.l_g[1] * loss_adv + args.l_g[2] * loss_ctc
                loss_g_total.backward()
                torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
                optimizer_g.step()

                # 累积 G 的损失和指标 (注意：这里是单次迭代的值)
                # 为了与原始代码的 epoch 平均方式对齐，我们将每次 G 训练的结果都记录下来
                # 这意味着每个 epoch 的记录次数会是原来的 k_train_g_per_d 倍
                epoch_loss_g_total.append(loss_g_total.item())
                epoch_loss_g_recon.append(loss_recon.item())
                epoch_loss_g_adv.append(loss_adv.item())
                epoch_loss_g_ctc.append(loss_ctc.item())
                epoch_acc_g_adv.append(acc_g_adv.item())

            # --- 训练判别器 1 次 ---
            for p in model_g.parameters():
                p.requires_grad_(False)  # 冻结 G
            for p in model_d.parameters():
                p.requires_grad_(True)  # 解冻 D

            optimizer_d.zero_grad()

            # 再次生成假样本 (因为 G 已经更新了 k 次)
            output = model_g(input)
            mel_out = DTW_align(output, target)

            # 真实样本
            real_valid, real_cl = model_d(target)
            # 生成样本
            fake_valid, fake_cl = model_d(mel_out.detach())

            # ---【新增】标签平滑 ---
            label_smoothing_eps = 0.2  # 一个常见的小值，例如 0.1 或 0.2
            smoothed_real_labels = torch.ones_like(valid) * (1.0 - label_smoothing_eps)  # 例如 0.9
            smoothed_fake_labels = torch.zeros_like(valid) + label_smoothing_eps  # 例如 0.1
            # ---【新增结束】---

            # # 判别器损失 components
            # loss_d_real_total = criterion_adv(real_valid, valid)  # D wants real to be classified as 1
            # loss_d_fake_total = criterion_adv(fake_valid, torch.zeros_like(valid))  # D wants fake to be classified as 0
            # loss_d_cl = criterion_cl(real_cl, target_cl)  # D wants to correctly classify real's label

            # 判别器损失 components
            # 【修改】使用平滑后的标签
            loss_d_real_total = criterion_adv(real_valid, smoothed_real_labels)  # D wants real to be classified as ~1.0
            # 【修改】使用平滑后的标签
            loss_d_fake_total = criterion_adv(fake_valid, smoothed_fake_labels)  # D wants fake to be classified as ~0.0
            loss_d_cl = criterion_cl(real_cl, target_cl)  # D wants to correctly classify real's label

            # 判别器总损失
            loss_d_total = args.l_d[0] * loss_d_cl + args.l_d[1] * 0.5 * (loss_d_real_total + loss_d_fake_total)
            loss_d_total.backward()
            optimizer_d.step()

            # 累积 D 的损失和指标 (注意：这里是单次迭代的值)
            # 与 G 类似，每个 epoch 的 D 记录次数是原始的次数
            epoch_loss_d_total.append(loss_d_total.item())
            epoch_loss_d_cl.append(loss_d_cl.item())
            epoch_loss_d_real.append(loss_d_real_total.item())  # L_adv for real
            epoch_loss_d_fake.append(loss_d_fake_total.item())  # L_adv for fake
            epoch_loss_d_real_total.append(
                0.5 * (loss_d_real_total.item() + loss_d_fake_total.item()))  # The adv part of D's loss

            acc_d_real = (real_valid.round() == valid).float().mean()
            acc_d_fake = (fake_valid.round() == torch.zeros_like(fake_valid)).float().mean()
            epoch_acc_d_real.append(acc_d_real.item())
            epoch_acc_d_fake.append(acc_d_fake.item())

            # 更新进度条后缀，显示最新的损失 (这里显示的是最后一次迭代的损失)
            # 如果想显示本轮（k次G + 1次D）的平均值，需要额外计算
            pbar.set_postfix({
                'G_Tot': f'{epoch_loss_g_total[-1] if epoch_loss_g_total else 0:.4f}',
                'G_Recon': f'{epoch_loss_g_recon[-1] if epoch_loss_g_recon else 0:.4f}',
                'G_Adv': f'{epoch_loss_g_adv[-1] if epoch_loss_g_adv else 0:.4f}',
                'G_CTC': f'{epoch_loss_g_ctc[-1] if epoch_loss_g_ctc else 0:.4f}',
                'D_Tot': f'{epoch_loss_d_total[-1] if epoch_loss_d_total else 0:.4f}',
                'D_CL': f'{epoch_loss_d_cl[-1] if epoch_loss_d_cl else 0:.4f}',
                'D_Real': f'{epoch_loss_d_real[-1] if epoch_loss_d_real else 0:.4f}',
                'D_Fake': f'{epoch_loss_d_fake[-1] if epoch_loss_d_fake else 0:.4f}',
                'Acc_G': f'{epoch_acc_g_adv[-1] if epoch_acc_g_adv else 0:.3f}',
                'Acc_D_R': f'{epoch_acc_d_real[-1] if epoch_acc_d_real else 0:.3f}',
                'Acc_D_F': f'{epoch_acc_d_fake[-1] if epoch_acc_d_fake else 0:.3f}'
            })

        # --- Epoch 结束处理 ---
        # 计算并打印本 Epoch 的平均损失
        # 由于G的记录次数是k倍，所以G的平均值会被稀释。如果要公平比较，可以只取每k次G训练的最后一次结果来平均，
        # 但最简单的方式是保留现有的平均逻辑，让其自然反映训练过程。
        avg_loss_g_total = np.mean(epoch_loss_g_total) if epoch_loss_g_total else 0
        avg_loss_g_recon = np.mean(epoch_loss_g_recon) if epoch_loss_g_recon else 0
        avg_loss_g_adv = np.mean(epoch_loss_g_adv) if epoch_loss_g_adv else 0
        avg_loss_g_ctc = np.mean(epoch_loss_g_ctc) if epoch_loss_g_ctc else 0

        avg_loss_d_total = np.mean(epoch_loss_d_total) if epoch_loss_d_total else 0
        avg_loss_d_cl = np.mean(epoch_loss_d_cl) if epoch_loss_d_cl else 0
        avg_loss_d_real = np.mean(epoch_loss_d_real) if epoch_loss_d_real else 0
        avg_loss_d_fake = np.mean(epoch_loss_d_fake) if epoch_loss_d_fake else 0
        avg_loss_d_adv_part = np.mean(epoch_loss_d_real_total) if epoch_loss_d_real_total else 0
        avg_acc_g_adv = np.mean(epoch_acc_g_adv) if epoch_acc_g_adv else 0
        avg_acc_d_real = np.mean(epoch_acc_d_real) if epoch_acc_d_real else 0
        avg_acc_d_fake = np.mean(epoch_acc_d_fake) if epoch_acc_d_fake else 0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f" Generator - Total: {avg_loss_g_total:.4f}, Recon: {avg_loss_g_recon:.4f}, Adv: {avg_loss_g_adv:.4f}, CTC: {avg_loss_g_ctc:.4f}, Acc_G: {avg_acc_g_adv:.3f}")
        print(
            f" Discriminator - Total: {avg_loss_d_total:.4f}, CL: {avg_loss_d_cl:.4f}, Real_Loss: {avg_loss_d_real:.4f}, Fake_Loss: {avg_loss_d_fake:.4f}, Adv_Part: {avg_loss_d_adv_part:.4f}")
        print(f" Accuracy - D_Real: {avg_acc_d_real:.3f}, D_Fake: {avg_acc_d_fake:.3f}")

        # 将详细损失写入 TensorBoard (同样，G的指标会被平均k次)
        writer.add_scalars('Loss/Generator', {
            'Total': avg_loss_g_total,
            'Reconstruction': avg_loss_g_recon,
            'Adversarial': avg_loss_g_adv,
            'CTC': avg_loss_g_ctc
        }, epoch)
        writer.add_scalars('Loss/Discriminator', {
            'Total': avg_loss_d_total,
            'Classification': avg_loss_d_cl,
            'Adv_Real_Fake': avg_loss_d_adv_part,
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Generator_Adv': avg_acc_g_adv,
            'Discriminator_Real': avg_acc_d_real,
            'Discriminator_Fake': avg_acc_d_fake
        }, epoch)

        # 学习率更新
        scheduler_g.step()
        scheduler_d.step()

        # 1. 每个 epoch 都进行验证，并检查是否保存最佳模型
        model_g.eval()
        model_d.eval()

        val_losses_g_recon = []  # Only recon loss for validation
        with torch.no_grad():
            for v_input, v_target, v_target_cl, v_voice, v_data_info in val_loader:
                v_input = v_input.to(device)
                v_target = v_target.to(device)

                v_out = model_g(v_input)
                v_mel = DTW_align(v_out, v_target)
                v_loss = criterion_recon(v_mel, v_target)
                val_losses_g_recon.append(v_loss.item())

        val_avg_loss_recon = np.mean(val_losses_g_recon)
        print(f" Validation Recon Loss: {val_avg_loss_recon:.4f}")
        writer.add_scalar('Loss/Validation_Recon', val_avg_loss_recon, epoch)

        # 保存最佳模型 (基于验证集上的重建损失)
        is_best = val_avg_loss_recon < best_loss
        if is_best:
            best_loss = val_avg_loss_recon
            state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
            state_d = {'state_dict': model_d.state_dict(), 'epoch': epoch}
            save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
            save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')
            print(f"✨ Best Model Saved (Val_Recon_Loss: {best_loss:.4f})!")

        # --- 验证与保存 ---
        if (epoch + 1) % args.val_interval == 0:
            # model_g.eval()
            # model_d.eval()
            #
            # val_losses_g_recon = []  # Only recon loss for validation
            # with torch.no_grad():
            #     for v_input, v_target, v_target_cl, v_voice, v_data_info in val_loader:
            #         v_input = v_input.to(device)
            #         v_target = v_target.to(device)
            #
            #         v_out = model_g(v_input)
            #         v_mel = DTW_align(v_out, v_target)
            #         v_loss = criterion_recon(v_mel, v_target)
            #         val_losses_g_recon.append(v_loss.item())
            #
            # val_avg_loss_recon = np.mean(val_losses_g_recon)
            # print(f" Validation Recon Loss: {val_avg_loss_recon:.4f}")
            # writer.add_scalar('Loss/Validation_Recon', val_avg_loss_recon, epoch)

            # # 保存最佳模型 (基于验证集上的重建损失)
            # is_best = val_avg_loss_recon < best_loss
            # if is_best:
            #     best_loss = val_avg_loss_recon
            #     state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
            #     state_d = {'state_dict': model_d.state_dict(), 'epoch': epoch}
            #     save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
            #     save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')
            #     print(f"✨ Best Model Saved (Val_Recon_Loss: {best_loss:.4f})!")

            # 保存音频样本
            saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch)

    print("🎉 Phase 2 Complete!")


def main(args):
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    args.device = device  # 将设备存入 args 以便后续使用

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
    args.writer = writer

    # ==========================
    # PHASE 1: 快速分类训练
    # ==========================
    print("\n🚀 Starting Phase 1 (Fast Classification)...")
    cls_model = run_fast_classification(args, device)

    if cls_model is None:
        print("❌ Phase 1 Failed. Exiting.")
        return

    del cls_model
    torch.cuda.empty_cache()

    # ==========================
    # PHASE 2: 语音生成训练
    # ==========================
    print("\n🎵 Starting Phase 2 (Mel Generation)...")
    run_mel_generation(args, device, writer)

    writer.close()
    print("🏁 All Training Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid EVRNet Training')
    # 基础参数
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000')
    parser.add_argument('--model_config', type=str, default='./models', help='Path to model config folder')
    parser.add_argument('--dataLoc', type=str, default='./processed_dataset_public')
    parser.add_argument('--config', type=str, default='./config_myPrivate1.json')
    parser.add_argument('--logDir', type=str, default='./TrainResult_models1_1_lowDis+dropout+noise+ch_init_downsample=24+dropout+1timeD2timeG+0.001advWeight+labelSmooth0.2+gdecay0.999+ddecay0.995_6')
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sub', type=str, default='sub'+str(subjects[0]))
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Voice_mel')

    # 训练参数
    parser.add_argument('--val_interval', type=int, default=5)
    args = parser.parse_args()
    main(args)