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
# ==============================
# 引入必要的模块
# ==============================

# 引入旧代码中的复杂模型 (确保 models/models.py 存在)
try:
    from models.models1_1_lowDis_4 import Discriminator, EVRNet_Generator
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
        # print(x.shape)
        # x = self.spatial_block1(x)
        x = self.spatial_block1(x.unsqueeze(1))  # (B, 32, 96, 24)
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


# ==============================
# 2. Phase 1: 数据处理与训练
# ==============================

word_to_label = {
    "my": 8, "dad": 0, "is": 1, "a": 2, "policeman": 3,
    "he": 4, "will": 5, "always": 6, "become": 7, "hero": 9
}


def process_file_fast(file_path, image_data, labels):
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
                image_data.append(segment_normalized.T)
    except FileNotFoundError:
        print(f"⚠️ Warning: File {file_path} not found. Skipping.")


def run_fast_classification(args, device):
    print("\n" + "=" * 50)
    print("🚀 Phase 1: Fast Classification (Old Logic)")
    print("=" * 50)
    image_data = []
    labels = []
    file_paths = ['vDLY-001_1.txt', 'vLCL-001_1.txt', 'vLHJ-001_1.txt', 'vLHJ-002_1.txt',
                  'vLCL-004_1.txt', 'vYHC02_1.txt', 'vLCL-01_1.txt', 'vLCL-02_1.txt',
                  'vCLB-001_1.txt', 'vYHC-001_1.txt', 'vLCL-003_1.txt']
    data_dir = "./orign_eeg_data/"
    for fp in file_paths:
        process_file_fast(os.path.join(data_dir, fp), image_data, labels)
    if not image_data:
        print("❌ No data loaded.")
        return None
    image_data_array = np.array(image_data)
    labels_array = np.array(labels)

    # 正常测试集
    # test_indices = [i for i in range(0, image_data_array.shape[0], 10)]
    # 跨被试测试集
    test_indices = [i for i in range(550, 605)]
    train_data = np.delete(image_data_array, test_indices, axis=0)

    test_data = image_data_array[test_indices]
    train_labels = np.delete(labels_array, test_indices, axis=0)
    test_labels = labels_array[test_indices]
    train_data = torch.tensor(train_data, dtype=torch.float32) # .unsqueeze(1)
    print(test_data[0])
    print(test_labels[0])
    print(test_data[-1])
    print(test_labels[-1])
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32) # .unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=args.batch_size, shuffle=False)

    num_classes = len(word_to_label)
    model = EVRNet_Classifier_Old(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc = 0.0
    epochs = 2000
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
            torch.save(model.state_dict(), os.path.join(args.savemodel, 'backbone_pretrained.pth'))
            print(f"✨ New Best Backbone Saved! (Acc: {best_acc:.2f}%)")
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


# def saveData(args, val_loader, models, epoch):
#     """保存生成的音频样本 (包括生成的音频和真实音频) - 修复版"""
#     model_g, _, vocoder, model_STT, decoder_STT = models
#     model_g.eval()
#     vocoder.eval()
#     model_STT.eval()
#
#     try:
#         input, target, target_cl, voice, data_info = next(iter(val_loader))
#     except StopIteration:
#         return
#
#     input = input.to(args.device)
#     target = target.to(args.device)
#     voice = torch.squeeze(voice, dim=-1).to(args.device)
#     labels = torch.argmax(target_cl, dim=1)
#
#     with torch.no_grad():
#         # ==========================
#         # 1. 处理生成样本 (Prediction)
#         # ==========================
#         output = model_g(input)
#
#         # 确保 target 和 output 在同一设备上
#         if output.is_cuda:
#             target = target.to(output.device)
#
#         # DTW 对齐
#         mel_out = DTW_align(output, target)
#         output_denorm = data_denorm(mel_out, data_info[0].to(args.device), data_info[1].to(args.device))
#
#         wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0],dim=0), vocoder, 1)
#
#         wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))
#
#         wav_recon = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)
#
#         if wav_recon.shape[1] !=  voice.shape[1]:
#             p = voice.shape[1] - wav_recon.shape[1]
#             p_s = p//2
#             p_e = p-p_s
#             wav_recon = F.pad(wav_recon, (p_s,p_e))
#
#         ##### STT Wav2Vec 2.0
#         gt_label = args.word_label[labels[0].item()]
#
#         transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, gt_label, 1)
#
#         # save
#         wav_recon = np.squeeze(wav_recon.cpu().detach().numpy())
#
#         str_tar = args.word_label[labels[0].item()].replace("|", ",")
#         str_tar = str_tar.replace(" ", ",")
#
#         str_pred = transcript_recon[0].replace("|", ",")
#         str_pred = str_pred.replace(" ", ",")
#
#         title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
#         wavio.write(args.savevoice + '/e{}_{}.wav'.format(str(str(epoch)), title), wav_recon, args.sample_rate_STT,
#                         sampwidth=1)
#
#         target = data_denorm(target, data_info[0], data_info[1])
#         gt_label = []
#         for k in range(len(target)):
#             gt_label.append(args.word_label[labels[k].item()])
#         wav_target = mel2wav_vocoder(target, vocoder, 1)
#         wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))
#         wav_target = torchaudio.functional.resample(wav_target, args.sampling_rate, args.sample_rate_STT)
#         if wav_target.shape[1] !=  voice.shape[1]:
#             p = voice.shape[1] - wav_target.shape[1]
#             p_s = p//2
#             p_e = p-p_s
#             wav_target = F.pad(wav_target, (p_s,p_e))
#         wav_target = wav_target.cpu().detach().numpy()
#         title = "Tar_{}".format(str_tar)
#         wavio.write(args.savevoice + "/" + title + ".wav", wav_target[0], args.sample_rate_STT, sampwidth=1)
#
#
#     # def saveData(args, val_loader, models, epoch):
#     """保存生成的音频样本 (包括生成的音频和真实音频)"""
#     model_g, _, vocoder, model_STT, decoder_STT = models
#     model_g.eval()
#     vocoder.eval()
#     model_STT.eval()
#
#     try:
#         input, target, target_cl, voice, data_info = next(iter(val_loader))
#     except StopIteration:
#         return
#
#     input = input.to(args.device)
#     voice = torch.squeeze(voice, dim=-1).to(args.device)
#     labels = torch.argmax(target_cl, dim=1)
#
#     with torch.no_grad():
#         # ==========================
#         # 1. 处理生成样本 (Prediction)
#         # ==========================
#         output = model_g(input)
#
#         # 【新增】确保 target 和 output 在同一设备上
#         if output.is_cuda:
#             target = target.to(output.device)
#
#         # DTW 对齐
#         mel_out = DTW_align(output, target)
#         output_denorm = data_denorm(mel_out, data_info[0].to(args.device), data_info[1].to(args.device))
#
#         # 生成音频 (取第一个样本)
#         wav_recon = vocoder(output_denorm[0:1])
#         if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
#             wav_recon = wav_recon.squeeze(1)
#
#         # 重采样
#         wav_recon = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)
#
#         # STT 识别
#         emission_recon, _ = model_STT(wav_recon)
#         transcript = decoder_STT(emission_recon[0])
#
#         # ==========================
#         # 2. 处理真实样本 (Ground Truth) - [新增部分]
#         # ==========================
#         # 对 target 进行反归一化
#         target_denorm = data_denorm(target[0:1], data_info[0].to(args.device), data_info[1].to(args.device))
#
#         # 通过 Vocoder 生成真实音频
#         wav_gt = vocoder(target_denorm)
#         if wav_gt.dim() == 3 and wav_gt.shape[1] == 1:
#             wav_gt = wav_gt.squeeze(1)
#
#         # 重采样以匹配 STT 采样率
#         wav_gt = torchaudio.functional.resample(wav_gt, args.sampling_rate, args.sample_rate_STT)
#
#         # ==========================
#         # 3. 保存文件
#         # ==========================
#         gt_label = args.word_label[labels[0].item()]
#         str_tar = gt_label.replace("|", "").replace(" ", "_")
#         str_pred = transcript.replace("|", "").replace(" ", "_")
#
#         # 文件名前缀
#         title_base = f"e{epoch}_Tar_{str_tar}"
#
#         # 保存生成的音频
#         title_pred = f"{title_base}-Pred_{str_pred}"[:100]
#         save_path_pred = os.path.join(args.savevoice, f"{title_pred}.wav")
#         wavio.write(save_path_pred, wav_recon.squeeze(0).cpu().numpy(), args.sample_rate_STT, sampwidth=1)
#         print(f"✅ Saved Predicted audio: {save_path_pred}")
#
#         # 保存真实音频 (Ground Truth)
#         title_gt = f"{title_base}-GT"[:100]
#         save_path_gt = os.path.join(args.savevoice, f"{title_gt}.wav")
#         wavio.write(save_path_gt, wav_gt.squeeze(0).cpu().numpy(), args.sample_rate_STT, sampwidth=1)
#         print(f"✅ Saved Ground Truth audio: {save_path_gt}")

# def saveData(args, val_loader, models, epoch):
#     """保存生成的音频样本"""
#     model_g, _, vocoder, model_STT, decoder_STT = models
#     model_g.eval()
#     vocoder.eval()
#     model_STT.eval()
#
#     try:
#         input, target, target_cl, voice, data_info = next(iter(val_loader))
#     except StopIteration:
#         return
#
#     input = input.to(args.device)
#     voice = torch.squeeze(voice, dim=-1).to(args.device)
#     labels = torch.argmax(target_cl, dim=1)
#
#     with torch.no_grad():
#         output = model_g(input)
#         # DTW 对齐
#         # 【新增】确保 target 和 output 在同一设备上
#         if output.is_cuda:
#             target = target.to(output.device)
#
#         mel_out = DTW_align(output, target)
#         output_denorm = data_denorm(mel_out, data_info[0].to(args.device), data_info[1].to(args.device))
#
#         # 生成音频 (取第一个样本)
#         wav_recon = vocoder(output_denorm[0:1])
#         if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
#             wav_recon = wav_recon.squeeze(1)
#
#         # 重采样
#         wav_recon = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)
#
#         # STT 识别
#         emission_recon, _ = model_STT(wav_recon)
#         transcript = decoder_STT(emission_recon[0])
#
#         # 保存
#         gt_label = args.word_label[labels[0].item()]
#         str_tar = gt_label.replace("|", "").replace(" ", "_")
#         str_pred = transcript.replace("|", "").replace(" ", "_")
#         title = f"Tar_{str_tar}-Pred_{str_pred}"[:100]
#
#         save_path = os.path.join(args.savevoice, f"e{epoch}_{title}.wav")
#         wavio.write(save_path, wav_recon.squeeze(0).cpu().numpy(), args.sample_rate_STT, sampwidth=2)
#         print(f"✅ Saved audio: {save_path}")

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

# def run_mel_generation(args, device, writer):
#     print("\n" + "=" * 50)
#     print("🎵 Phase 2: Mel Generation (Complex GAN Logic)")
#     print("=" * 50)
#
#     # 1. 加载配置
#     with open(args.config) as f:
#         config = json.load(f)
#         for k, v in config.items():
#             setattr(args, k, v)
#
#     # 设置设备属性以便后续使用
#     args.device = device
#
#     # 2. 初始化 STT Bundle
#     bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
#     args.sample_rate_STT = bundle.sample_rate
#     if not hasattr(args, 'word_index'):
#         args.word_index, args.word_length = word_index(args.word_label, bundle)
#
#     # 3. 加载 Vocoder
#     config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
#     with open(config_file) as f:
#         h = AttrDict(json.load(f))
#     vocoder = model_HiFi(h).to(device)
#     vocoder.load_state_dict(torch.load(args.vocoder_pre)['generator'])
#     vocoder.eval()
#     for p in vocoder.parameters(): p.requires_grad = False
#
#     # 4. 加载 STT
#     model_STT = bundle.get_model().to(device)
#     decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
#     model_STT.eval()
#     for p in model_STT.parameters(): p.requires_grad = False
#
#     # 5. 初始化生成器和判别器 (使用旧代码的复杂结构)
#     try:
#         config_file_g = os.path.join(args.model_config, 'config_g.json')
#         with open(config_file_g) as f:
#             h_g = AttrDict(json.load(f))
#         if not hasattr(h_g, 'upsample_initial_channel'):
#             h_g.upsample_initial_channel = 512
#
#         model_g = EVRNet_Generator(h_g).to(device)
#         config_file_d = os.path.join(args.model_config, 'config_d.json')
#         with open(config_file_d) as f:
#             h_d = AttrDict(json.load(f))
#         model_d = Discriminator(h_d).to(device)
#         print("✅ Loaded Complex Generator & Discriminator from config.")
#     except Exception as e:
#         print(f"❌ Error loading Generator/Discriminator config: {e}")
#         print("⚠️ Please ensure 'models/config_g.json' and 'models/config_d.json' exist.")
#         return
#
#     # 6. 加载 Phase 1 权重到 Generator
#     bp_path = os.path.join(args.savemodel, 'backbone_pretrained.pth')
#     if os.path.exists(bp_path):
#         cls_state_dict = torch.load(bp_path, map_location=device)
#         model_g_dict = model_g.state_dict()
#         pretrained_dict = {k: v for k, v in cls_state_dict.items() if k != 'fc_layer.weight' and k != 'fc_layer.bias'}
#         model_g_dict.update(pretrained_dict)
#         model_g.load_state_dict(model_g_dict, strict=False)
#         print("✅ 完美加载！生成器成功继承了分类器的 Backbone + Pool + Mamba 权重")
#
#     # 7. 数据加载
#     try:
#         from NeuroTalkDataset import myDataset
#         trainset = myDataset(mode=0, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
#         valset = myDataset(mode=2, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
#         train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#         val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
#     except ImportError:
#         print("❌ NeuroTalkDataset not found.")
#         return
#
#     # 8. 优化器与损失函数
#     optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
#     optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)
#
#     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
#     scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)
#
#     criterion_recon = RMSELoss().to(device)
#     criterion_adv = nn.BCELoss().to(device)
#     criterion_ctc = nn.CTCLoss().to(device)
#     criterion_cl = nn.CrossEntropyLoss().to(device)
#     CER = CharErrorRate().to(device)
#
#     best_loss = 1000
#
#     # ==========================
#     # 训练循环
#     # ==========================
#     for epoch in range(args.max_epochs):
#         model_g.train()
#         model_d.train()
#         vocoder.eval()
#         model_STT.eval()
#
#         # 初始化所有需要记录的详细损失列表
#         epoch_loss_g_total = []
#         epoch_loss_g_recon = []
#         epoch_loss_g_adv = []
#         epoch_loss_g_ctc = []
#
#         epoch_loss_d_total = []
#         epoch_loss_d_cl = []
#         epoch_loss_d_real = []
#         epoch_loss_d_fake = []
#         epoch_loss_d_real_total = []  # L_D(real, fake)
#
#         epoch_acc_g_adv = []
#         epoch_acc_d_real = []
#         epoch_acc_d_fake = []
#
#         pbar = tqdm(train_loader, desc=f"Gen Ep {epoch + 1}/{args.max_epochs}")
#
#         for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
#             input = input.to(device)
#             target = target.to(device)
#             target_cl = target_cl.to(device)
#             voice = torch.squeeze(voice, dim=-1).to(device)
#             labels = torch.argmax(target_cl, dim=1)
#
#             # 准备 CTC 标签
#             gt_label_idx = []
#             gt_length = []
#             for j in range(len(labels)):
#                 gt_label_idx.append(args.word_index[labels[j].item()])
#                 gt_length.append(args.word_length[labels[j].item()])
#             gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64).to(device)
#             gt_length = torch.tensor(gt_length, dtype=torch.int64).to(device)
#
#             # --- 训练生成器 ---
#             for p in model_g.parameters(): p.requires_grad_(True)
#             for p in model_d.parameters(): p.requires_grad_(False)
#             optimizer_g.zero_grad()
#
#             output = model_g(input)
#             mel_out = DTW_align(output, target)
#
#             # 判别器输出 (用于 GAN Loss)
#             g_valid, _ = model_d(mel_out)
#             valid = torch.ones((len(input), 1), dtype=torch.float32).to(device)
#
#             # 1. 重建损失 (RMSE)
#             loss_recon = criterion_recon(mel_out, target)
#
#             # 2. GAN 损失 (Generator perspective)
#             loss_adv = criterion_adv(g_valid, valid)
#             acc_g_adv = (g_valid.round() == valid).float().mean()
#
#             # 3. CTC 损失 (通过 Vocoder -> STT)
#             with torch.no_grad():
#                 output_denorm = data_denorm(mel_out, data_info[0].to(device), data_info[1].to(device))
#
#             wav_recon = vocoder(output_denorm)
#             if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
#                 wav_recon = wav_recon.squeeze(1)
#
#             voice_for_stt = torchaudio.functional.resample(voice, args.sampling_rate, args.sample_rate_STT)
#             wav_recon_stt = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)
#
#             max_len = max(voice_for_stt.shape[1], wav_recon_stt.shape[1])
#             voice_for_stt = F.pad(voice_for_stt, (0, max_len - voice_for_stt.shape[1]))
#             wav_recon_stt = F.pad(wav_recon_stt, (0, max_len - wav_recon_stt.shape[1]))
#
#             emission_recon, _ = model_STT(wav_recon_stt)
#             emission_recon_ = emission_recon.log_softmax(2)
#
#             input_lengths = torch.full(size=(emission_recon.size(0),), fill_value=emission_recon.size(1),
#                                        dtype=torch.long).to(device)
#             loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)
#
#             # 生成器总损失
#             loss_g_total = args.l_g[0] * loss_recon + args.l_g[1] * loss_adv + args.l_g[2] * loss_ctc
#
#             loss_g_total.backward()
#             torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
#             optimizer_g.step()
#
#             # --- 训练判别器 ---
#             for p in model_g.parameters(): p.requires_grad_(False)
#             for p in model_d.parameters(): p.requires_grad_(True)
#             optimizer_d.zero_grad()
#
#             # 真实样本
#             real_valid, real_cl = model_d(target)
#             # 生成样本
#             fake_valid, fake_cl = model_d(mel_out.detach())
#
#             # 判别器损失 components
#             loss_d_real_total = criterion_adv(real_valid, valid)  # D wants real to be classified as 1
#             loss_d_fake_total = criterion_adv(fake_valid, torch.zeros_like(valid))  # D wants fake to be classified as 0
#             loss_d_cl = criterion_cl(real_cl, target_cl)  # D wants to correctly classify real's label
#
#             # 判别器总损失
#             loss_d_total = args.l_d[0] * loss_d_cl + args.l_d[1] * 0.5 * (loss_d_real_total + loss_d_fake_total)
#
#             loss_d_total.backward()
#             optimizer_d.step()
#
#             # 记录所有详细指标到各自的列表
#             epoch_loss_g_total.append(loss_g_total.item())
#             epoch_loss_g_recon.append(loss_recon.item())
#             epoch_loss_g_adv.append(loss_adv.item())
#             epoch_loss_g_ctc.append(loss_ctc.item())
#
#             epoch_acc_g_adv.append(acc_g_adv.item())
#
#             epoch_loss_d_total.append(loss_d_total.item())
#             epoch_loss_d_cl.append(loss_d_cl.item())
#             epoch_loss_d_real.append(loss_d_real_total.item())  # L_adv for real
#             epoch_loss_d_fake.append(loss_d_fake_total.item())  # L_adv for fake
#             epoch_loss_d_real_total.append(
#                 0.5 * (loss_d_real_total.item() + loss_d_fake_total.item()))  # The adv part of D's loss
#
#             acc_d_real = (real_valid.round() == valid).float().mean()
#             acc_d_fake = (fake_valid.round() == torch.zeros_like(fake_valid)).float().mean()
#             epoch_acc_d_real.append(acc_d_real.item())
#             epoch_acc_d_fake.append(acc_d_fake.item())
#
#             # 更新进度条后缀，显示更详细的损失
#             pbar.set_postfix({
#                 'G_Tot': f'{np.mean(epoch_loss_g_total):.4f}',
#                 'G_Recon': f'{np.mean(epoch_loss_g_recon):.4f}',
#                 'G_Adv': f'{np.mean(epoch_loss_g_adv):.4f}',
#                 'G_CTC': f'{np.mean(epoch_loss_g_ctc):.4f}',
#                 'D_Tot': f'{np.mean(epoch_loss_d_total):.4f}',
#                 'D_CL': f'{np.mean(epoch_loss_d_cl):.4f}',
#                 'D_Real': f'{np.mean(epoch_loss_d_real):.4f}',
#                 'D_Fake': f'{np.mean(epoch_loss_d_fake):.4f}',
#                 'Acc_G': f'{np.mean(epoch_acc_g_adv):.3f}',
#                 'Acc_D_R': f'{np.mean(epoch_acc_d_real):.3f}',
#                 'Acc_D_F': f'{np.mean(epoch_acc_d_fake):.3f}'
#             })
#
#         # --- Epoch 结束处理 ---
#         # 计算并打印本 Epoch 的平均损失
#         avg_loss_g_total = np.mean(epoch_loss_g_total)
#         avg_loss_g_recon = np.mean(epoch_loss_g_recon)
#         avg_loss_g_adv = np.mean(epoch_loss_g_adv)
#         avg_loss_g_ctc = np.mean(epoch_loss_g_ctc)
#
#         avg_loss_d_total = np.mean(epoch_loss_d_total)
#         avg_loss_d_cl = np.mean(epoch_loss_d_cl)
#         avg_loss_d_real = np.mean(epoch_loss_d_real)
#         avg_loss_d_fake = np.mean(epoch_loss_d_fake)
#         avg_loss_d_adv_part = np.mean(epoch_loss_d_real_total)  # The adversarial part of D's loss
#
#         avg_acc_g_adv = np.mean(epoch_acc_g_adv)
#         avg_acc_d_real = np.mean(epoch_acc_d_real)
#         avg_acc_d_fake = np.mean(epoch_acc_d_fake)
#
#         print(f"\nEpoch {epoch + 1} Summary:")
#         print(
#             f"  Generator - Total: {avg_loss_g_total:.4f}, Recon: {avg_loss_g_recon:.4f}, Adv: {avg_loss_g_adv:.4f}, CTC: {avg_loss_g_ctc:.4f}, Acc_G: {avg_acc_g_adv:.3f}")
#         print(
#             f"  Discriminator - Total: {avg_loss_d_total:.4f}, CL: {avg_loss_d_cl:.4f}, Real_Loss: {avg_loss_d_real:.4f}, Fake_Loss: {avg_loss_d_fake:.4f}, Adv_Part: {avg_loss_d_adv_part:.4f}")
#         print(f"  Accuracy - D_Real: {avg_acc_d_real:.3f}, D_Fake: {avg_acc_d_fake:.3f}")
#
#         # 将详细损失写入 TensorBoard
#         writer.add_scalars('Loss/Generator', {
#             'Total': avg_loss_g_total,
#             'Reconstruction': avg_loss_g_recon,
#             'Adversarial': avg_loss_g_adv,
#             'CTC': avg_loss_g_ctc
#         }, epoch)
#         writer.add_scalars('Loss/Discriminator', {
#             'Total': avg_loss_d_total,
#             'Classification': avg_loss_d_cl,
#             'Adv_Real_Fake': avg_loss_d_adv_part,  # Shows the adversarial training balance
#         }, epoch)
#         writer.add_scalars('Accuracy', {
#             'Generator_Adv': avg_acc_g_adv,
#             'Discriminator_Real': avg_acc_d_real,
#             'Discriminator_Fake': avg_acc_d_fake
#         }, epoch)
#
#         # 学习率更新
#         scheduler_g.step()
#         scheduler_d.step()
#
#         # --- 验证与保存 ---
#         if (epoch + 1) % args.val_interval == 0:
#             model_g.eval()
#             model_d.eval()
#
#             val_losses_g_recon = []  # Only recon loss for validation
#             with torch.no_grad():
#                 for v_input, v_target, v_target_cl, v_voice, v_data_info in val_loader:
#                     v_input = v_input.to(device)
#                     v_target = v_target.to(device)
#
#                     v_out = model_g(v_input)
#                     v_mel = DTW_align(v_out, v_target)
#                     v_loss = criterion_recon(v_mel, v_target)
#                     val_losses_g_recon.append(v_loss.item())
#
#             val_avg_loss_recon = np.mean(val_losses_g_recon)
#             print(f"  Validation Recon Loss: {val_avg_loss_recon:.4f}")
#             writer.add_scalar('Loss/Validation_Recon', val_avg_loss_recon, epoch)
#
#             # 保存最佳模型 (基于验证集上的重建损失)
#             is_best = val_avg_loss_recon < best_loss
#             if is_best:
#                 best_loss = val_avg_loss_recon
#                 state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
#                 state_d = {'state_dict': model_d.state_dict(), 'epoch': epoch}
#                 save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
#                 save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')
#                 print(f"✨ Best Model Saved (Val_Recon_Loss: {best_loss:.4f})!")
#
#             # 保存音频样本
#             saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch)
#
#     print("🎉 Phase 2 Complete!")

# def run_mel_generation(args, device, writer):
#     print("\n" + "=" * 50)
#     print("🎵 Phase 2: Mel Generation (Complex GAN Logic)")
#     print("=" * 50)
#
#     # 1. 加载配置
#     with open(args.config) as f:
#         config = json.load(f)
#         for k, v in config.items():
#             setattr(args, k, v)
#
#     # 设置设备属性以便后续使用
#     args.device = device
#
#     # 2. 初始化 STT Bundle
#     bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
#     args.sample_rate_STT = bundle.sample_rate
#     if not hasattr(args, 'word_index'):
#         args.word_index, args.word_length = word_index(args.word_label, bundle)
#
#     # 3. 加载 Vocoder
#     config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
#     with open(config_file) as f:
#         h = AttrDict(json.load(f))
#     vocoder = model_HiFi(h).to(device)
#     vocoder.load_state_dict(torch.load(args.vocoder_pre)['generator'])
#     vocoder.eval()
#     for p in vocoder.parameters(): p.requires_grad = False
#
#     # 4. 加载 STT
#     model_STT = bundle.get_model().to(device)
#     decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
#     model_STT.eval()
#     for p in model_STT.parameters(): p.requires_grad = False
#
#     # 5. 初始化生成器和判别器 (使用旧代码的复杂结构)
#     # 注意：这里需要 Generator 和 Discriminator 的配置文件
#     # 假设配置文件在 args.model_config 目录下，或者我们手动构建一个简单的配置
#     # 如果找不到 config_g.json，这里可能需要你手动指定路径
#     try:
#         config_file_g = os.path.join(args.model_config, 'config_g.json')
#         with open(config_file_g) as f:
#             h_g = AttrDict(json.load(f))
#         # 确保 h 中有 upsample_initial_channel，如果没有则手动指定
#         if not hasattr(h_g, 'upsample_initial_channel'):
#             h_g.upsample_initial_channel = 512
#
#         model_g = EVRNet_Generator(h_g).to(device)
#         # model_g = Generator(h_g).to(device)
#
#         config_file_d = os.path.join(args.model_config, 'config_d.json')
#         with open(config_file_d) as f:
#             h_d = AttrDict(json.load(f))
#         model_d = Discriminator(h_d).to(device)
#         print("✅ Loaded Complex Generator & Discriminator from config.")
#     except Exception as e:
#         print(f"❌ Error loading Generator/Discriminator config: {e}")
#         print("⚠️ Please ensure 'models/config_g.json' and 'models/config_d.json' exist.")
#         return
#
#     # 6. 加载 Phase 1 权重到 Generator
#     bp_path = os.path.join(args.savemodel, 'backbone_pretrained.pth')
#     if os.path.exists(bp_path):
#         cls_state_dict = torch.load(bp_path, map_location=device)
#         model_g_dict = model_g.state_dict()
#
#         # 1. 过滤掉分类器特有的 FC 层
#         pretrained_dict = {k: v for k, v in cls_state_dict.items() if k != 'fc_layer.weight' and k != 'fc_layer.bias'}
#
#         # 2. 更新生成器字典
#         # 因为生成器里也有 backbone, avg_pooling, mamba，所以 key 是一样的，直接覆盖
#         model_g_dict.update(pretrained_dict)
#
#         # 3. 加载
#         model_g.load_state_dict(model_g_dict, strict=False)  # strict=False 是为了忽略新增的 mamba_to_gen_proj 和 ups 层
#
#         print("✅ 完美加载！生成器成功继承了分类器的 Backbone + Pool + Mamba 权重")
#
#     # 7. 数据加载
#     try:
#         from NeuroTalkDataset import myDataset
#         trainset = myDataset(mode=0, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
#         valset = myDataset(mode=2, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
#         train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#         val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
#     except ImportError:
#         print("❌ NeuroTalkDataset not found.")
#         return
#
#     # 8. 优化器与损失函数
#     optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
#     optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)
#
#     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
#     scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)
#
#     criterion_recon = RMSELoss().to(device)
#     criterion_adv = nn.BCELoss().to(device)
#     criterion_ctc = nn.CTCLoss().to(device)
#     criterion_cl = nn.CrossEntropyLoss().to(device)
#     CER = CharErrorRate().to(device)
#
#     best_loss = 1000
#
#     # ==========================
#     # 训练循环
#     # ==========================
#     for epoch in range(args.max_epochs):
#         model_g.train()
#         model_d.train()
#         vocoder.eval()
#         model_STT.eval()
#
#         epoch_loss_g = []
#         epoch_loss_d = []
#         epoch_acc_g = []
#         epoch_acc_d = []
#
#         pbar = tqdm(train_loader, desc=f"Gen Ep {epoch + 1}/{args.max_epochs}")
#
#         for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
#             input = input.to(device)
#             # print(input.shape)
#             # print('**********************************************')
#             target = target.to(device)
#             target_cl = target_cl.to(device)
#             voice = torch.squeeze(voice, dim=-1).to(device)
#             labels = torch.argmax(target_cl, dim=1)
#
#             # 准备 CTC 标签
#             gt_label_idx = []
#             gt_length = []
#             for j in range(len(labels)):
#                 gt_label_idx.append(args.word_index[labels[j].item()])
#                 gt_length.append(args.word_length[labels[j].item()])
#             gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64).to(device)
#             gt_length = torch.tensor(gt_length, dtype=torch.int64).to(device)
#
#             # --- 训练生成器 ---
#             for p in model_g.parameters(): p.requires_grad_(True)
#             for p in model_d.parameters(): p.requires_grad_(False)
#             optimizer_g.zero_grad()
#
#             output = model_g(input)
#             # print(output.shape)
#             # print(target.shape)
#             # print('===============================')
#             # DTW 对齐
#             mel_out = DTW_align(output, target)
#
#             # 判别器输出 (用于 GAN Loss)
#             g_valid, _ = model_d(mel_out)
#             valid = torch.ones((len(input), 1), dtype=torch.float32).to(device)
#
#             # 1. 重建损失 (RMSE)
#             loss_recon = criterion_recon(mel_out, target)
#
#             # 2. GAN 损失
#             loss_valid = criterion_adv(g_valid, valid)
#             acc_g_valid = (g_valid.round() == valid).float().mean()
#
#             # 3. CTC 损失 (通过 Vocoder -> STT)
#             with torch.no_grad():
#                 output_denorm = data_denorm(mel_out, data_info[0].to(device), data_info[1].to(device))
#
#             # Vocoder 生成
#             wav_recon = vocoder(output_denorm)
#             if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
#                 wav_recon = wav_recon.squeeze(1)
#
#             # 重采样以匹配 STT (16kHz)
#             voice_for_stt = torchaudio.functional.resample(voice, args.sampling_rate, args.sample_rate_STT)
#             wav_recon_stt = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)
#
#             # 填充对齐
#             max_len = max(voice_for_stt.shape[1], wav_recon_stt.shape[1])
#             voice_for_stt = F.pad(voice_for_stt, (0, max_len - voice_for_stt.shape[1]))
#             wav_recon_stt = F.pad(wav_recon_stt, (0, max_len - wav_recon_stt.shape[1]))
#
#             # STT 前向传播
#             emission_recon, _ = model_STT(wav_recon_stt)
#             emission_recon_ = emission_recon.log_softmax(2)
#
#             input_lengths = torch.full(size=(emission_recon.size(0),), fill_value=emission_recon.size(1),
#                                        dtype=torch.long).to(device)
#             loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)
#
#             # 生成器总损失
#             loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid + args.l_g[2] * loss_ctc
#
#             loss_g.backward()
#             torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
#             optimizer_g.step()
#
#             # --- 训练判别器 ---
#             for p in model_g.parameters(): p.requires_grad_(False)
#             for p in model_d.parameters(): p.requires_grad_(True)
#             optimizer_d.zero_grad()
#
#             # 真实样本
#             real_valid, real_cl = model_d(target)
#             # 生成样本
#             fake_valid, fake_cl = model_d(mel_out.detach())
#
#             # 判别器损失
#             loss_d_real = criterion_adv(real_valid, valid)
#             loss_d_fake = criterion_adv(fake_valid, torch.zeros_like(valid))
#             loss_d_cl = criterion_cl(real_cl, target_cl)
#
#             loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * 0.5 * (loss_d_real + loss_d_fake)
#
#             loss_d.backward()
#             optimizer_d.step()
#
#             # 记录指标
#             epoch_loss_g.append([loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item()])
#             epoch_loss_d.append([loss_d.item(), loss_d_cl.item(), loss_d_real.item(), loss_d_fake.item()])
#
#             acc_d_real = (real_valid.round() == valid).float().mean()
#             acc_d_fake = (fake_valid.round() == torch.zeros_like(valid)).float().mean()
#             epoch_acc_g.append([acc_g_valid.item()])
#             epoch_acc_d.append([acc_d_real.item(), acc_d_fake.item()])
#
#             pbar.set_postfix({
#                 'G_Loss': f'{np.mean([x[0] for x in epoch_loss_g]):.4f}',
#                 'D_Loss': f'{np.mean([x[0] for x in epoch_loss_d]):.4f}'
#             })
#
#             # --- Epoch 结束处理 ---
#         avg_loss_g = np.mean([x[0] for x in epoch_loss_g])
#         avg_loss_d = np.mean([x[0] for x in epoch_loss_d])
#
#         print(f"Epoch {epoch + 1} - G_Loss: {avg_loss_g:.4f}, D_Loss: {avg_loss_d:.4f}")
#
#         # 学习率更新
#         scheduler_g.step()
#         scheduler_d.step()
#
#         # --- 验证与保存 ---
#         if (epoch + 1) % args.val_interval == 0:
#             model_g.eval()
#             model_d.eval()
#
#             # 简单的验证循环 (计算平均 Loss)
#             val_losses_g = []
#             with torch.no_grad():
#                 for v_input, v_target, v_target_cl, v_voice, v_data_info in val_loader:
#                     v_input = v_input.to(device)
#                     v_target = v_target.to(device)
#                     v_labels = torch.argmax(v_target_cl, dim=1).to(device)
#
#                     v_out = model_g(v_input)
#                     v_mel = DTW_align(v_out, v_target)
#
#                     # 仅计算重建损失作为验证指标
#                     v_loss = criterion_recon(v_mel, v_target)
#                     val_losses_g.append(v_loss.item())
#
#             val_avg_loss = np.mean(val_losses_g)
#             print(f"Validation Loss: {val_avg_loss:.4f}")
#
#             # 保存最佳模型
#             is_best = val_avg_loss < best_loss
#             if is_best:
#                 best_loss = val_avg_loss
#                 state_g = {'state_dict': model_g.state_dict(), 'epoch': epoch}
#                 state_d = {'state_dict': model_d.state_dict(), 'epoch': epoch}
#                 save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
#                 save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')
#                 print(f"✨ Best Model Saved!")
#
#             # 保存音频样本
#             saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch)
#
#     print("🎉 Phase 2 Complete!")


# ==============================
# 4. 主程序入口
# ==============================

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

    # # ==========================
    # # PHASE 1: 快速分类训练
    # # ==========================
    # print("\n🚀 Starting Phase 1 (Fast Classification)...")
    # cls_model = run_fast_classification(args, device)
    #
    # if cls_model is None:
    #     print("❌ Phase 1 Failed. Exiting.")
    #     return
    #
    # del cls_model
    # torch.cuda.empty_cache()

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
    parser.add_argument('--dataLoc', type=str, default='./processed_dataset_1')
    parser.add_argument('--config', type=str, default='./config_myPrivate1.json')
    parser.add_argument('--logDir', type=str, default='./6_noclassPre')
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Voice_mel')

    # 训练参数
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=5)

    # 标签
    # parser.add_argument('--word_label', type=list,
    #                     default=["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "hero"])

    # parser.add_argument('--sampling_rate', type=int, default=22050, help='梅尔频谱图的采样率，需要与生成器输入匹配')

    args = parser.parse_args()
    main(args)