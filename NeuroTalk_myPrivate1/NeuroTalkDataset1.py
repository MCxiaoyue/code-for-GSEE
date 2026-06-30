import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json

epsilon = np.finfo(float).eps


class myDataset(Dataset):
    def __init__(self, mode, data="./", task="SpokenEEG", recon="Y_mel"):
        self.sample_rate = 8000
        self.n_classes = 13
        self.mode = mode
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0

        # 统计文件数量
        self.lenth = len(os.listdir(self.savedata + '/train/Y/'))
        self.lenthtest = len(os.listdir(self.savedata + '/test/Y/'))
        self.lenthval = len(os.listdir(self.savedata + '/val/Y/'))

        # === 新增：加载全局统计量 ===
        stats_path = os.path.join(self.savedata, 'global_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.GLOBAL_MEAN = float(stats['mel_global_mean'])
            self.GLOBAL_STD = float(stats['mel_global_std'])
            print(f"[Dataset] 加载全局统计量 -> Mean: {self.GLOBAL_MEAN:.4f}, Std: {self.GLOBAL_STD:.4f}")
        #     # 防止除以零
        #     if self.GLOBAL_STD < 1e-6:
        #         print("[警告] 全局标准差过小，设为 1.0")
        #         self.GLOBAL_STD = 1.0
        # else:
        #     # 如果没有找到文件，回退到旧逻辑或报错 (建议报错以避免训练推理不一致)
        #     print(f"[警告] 未找到 {stats_path}，将使用默认值 (可能导致重构噪声)。请运行预处理脚本生成该文件。")
        #     self.GLOBAL_MEAN = -4.5  # 经验默认值
        #     self.GLOBAL_STD = 1.0
        # # ===========================

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        if self.mode == 2:
            forder_name = self.savedata + '/val/'
        elif self.mode == 1:
            forder_name = self.savedata + '/test/'
        else:
            forder_name = self.savedata + '/train/'

        # 1. 读取 Input (EEG)
        allFileList = os.listdir(forder_name + self.task + "/")
        allFileList.sort()
        file_name = forder_name + self.task + '/' + allFileList[idx]

        if self.task.find('mel') != -1:
            input, avg_input, std_input = self.read_data(file_name)
        elif self.task.find('Voice') != -1:
            input, avg_input, std_input = self.read_voice_data(file_name)
        else:
            input, avg_input, std_input = self.read_data(file_name)

        # 2. 读取 Target (Mel)
        allFileList = os.listdir(forder_name + self.recon + "/")
        allFileList.sort()
        file_name = forder_name + self.recon + '/' + allFileList[idx]

        if self.recon.find('mel') != -1:
            # 关键：read_data 现在会使用全局统计量进行归一化
            target, avg_target, std_target = self.read_data(file_name)
        elif self.recon.find('Voice') != -1:
            target, avg_target, std_target = self.read_voice_data(file_name)
        else:
            target, avg_target, std_target = self.read_data(file_name)

            # 3. 读取 Voice (用于参考，通常不归一化或单独处理)
        allFileList = os.listdir(forder_name + "Voice/")
        allFileList.sort()
        file_name = forder_name + "Voice/" + allFileList[idx]
        voice, _, _ = self.read_voice_data(file_name)

        # 4. 读取 Label
        allFileList = os.listdir(forder_name + "Y/")
        allFileList.sort()
        file_name = forder_name + 'Y/' + allFileList[idx]
        target_cl, _, _ = self.read_raw_data(file_name)
        target_cl = np.squeeze(target_cl)

        # 转 Tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # 返回的 data_info 现在包含的是 GLOBAL_MEAN 和 GLOBAL_STD
        # 这样在 eval.py 中反归一化时，就能正确还原到 HiFi-GAN 期望的空间
        return input, target, target_cl, voice, (avg_target, std_target, avg_input, std_input)

    # ... read_vector_data 和 read_voice_data 保持不变 ...
    def read_voice_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        data = np.array(data / self.max_audio).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        return data, avg, self.max_audio

    def read_raw_data(self, file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)
        data = np.array(data).astype(np.float32)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)
        return data, avg, std

    # === 修改核心：read_data ===
    def read_data(self, file_name):
        """
        读取 CSV 并使用【全局统计量】进行 Z-Score 归一化。
        不再使用每样本的 Min-Max。
        """
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        data = np.array(data).astype(np.float32)

        # 旧逻辑 (已废弃):
        # max_ = np.max(data)
        # min_ = np.min(data)
        # avg = (max_ + min_) / 2
        # std = (max_ - min_) / 2

        # 新逻辑：使用全局统计量
        avg = self.GLOBAL_MEAN
        std = self.GLOBAL_STD

        # Z-Score 归一化: (x - mean) / std
        # 注意：CSV 中保存的是 Raw Log-Mel，所以这里直接减全局均值除全局标准差
        data_norm = (data - avg) / std

        return data_norm, avg, std
