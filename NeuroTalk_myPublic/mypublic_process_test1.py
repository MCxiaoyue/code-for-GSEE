import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import json
from torch.utils.data import TensorDataset
from mne import read_epochs
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =================配置区域=================
# --- 通用路径配置 ---

# --- 公共数据集参数 ---
# 注意：EEG采样率是1024，但音频和Mel是基于HiFi-GAN配置的采样率（通常22050）
TASK = 'audio'
TAG = 's'
DURATION = 2  # 秒
SUBJECTS = ['19']  # 可以扩展为 ['8', '12', '20', ...]
SESSIONS = ['1']

# 注意：这里保留了原脚本的输出结构，但输入路径改为公共数据集路径
ORIG_WAV_DIR = "./orign_wav_public_resampled"  # 公共数据集音频路径
OUTPUT_ROOT = "./processed_dataset_public/sub"+str(SUBJECTS[0])  # 输出路径改为公共数据集专用

# --- HiFi-GAN 配置 ---
HIFIGAN_CONFIG_PATH = './pretrained_model/UNIVERSAL_V1/config.json'
HIFIGAN_VOCODER_PATH = './pretrained_model/UNIVERSAL_V1/g_02500000'

# --- 其他参数 ---
WORDS_SEQ = ["flower", "penguin", "guitar"]  # 根据你的公共数据集类别定义
WORD_TO_LABEL = {"flower": 0, "penguin": 1, "guitar": 2}
NUM_CLASSES = 3
MAX_WAV_VALUE = 32768

# --- 全局变量 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mel_transform = None
hifigan_config = None


# =========================================

def setup_hifigan():
    """初始化 HiFi-GAN 配置和 Mel 变换器"""
    global hifigan_config, mel_transform, device
    global TARGET_SR_AUDIO  # 需要声明为全局，以便在 load_audio_templates 中使用

    if not os.path.exists(HIFIGAN_CONFIG_PATH):
        raise FileNotFoundError(f"未找到 HiFi-GAN 配置文件：{HIFIGAN_CONFIG_PATH}")

    print("正在加载 HiFi-GAN 配置...")
    with open(HIFIGAN_CONFIG_PATH, 'r') as f:
        h_dict = json.load(f)

    hifigan_config = h_dict  # 简单使用字典
    TARGET_SR_AUDIO = hifigan_config['sampling_rate']  # 从配置中读取音频采样率
    print(f"HiFi-GAN 采样率设定为：{TARGET_SR_AUDIO}")

    # 定义 Mel 变换器
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SR_AUDIO,
        n_fft=hifigan_config['n_fft'],
        win_length=hifigan_config['win_size'],
        hop_length=hifigan_config['hop_size'],
        f_min=hifigan_config['fmin'],
        f_max=hifigan_config['fmax'],
        n_mels=hifigan_config['num_mels'],
        power=1,
        normalized=False,
        norm='slaney',
        mel_scale='slaney'
    ).to(device)
    mel_transform.eval()


def create_directories():
    splits = ['train', 'test', 'val']
    subfolders = ['SpokenEEG', 'Voice', 'Voice_mel', 'Y']
    for split in splits:
        for sub in subfolders:
            path = os.path.join(OUTPUT_ROOT, split, sub)
            os.makedirs(path, exist_ok=True)
    print("目录结构创建完成。")


def load_audio_templates():
    """
    根据WORDS_SEQ加载音频文件。
    注意：公共数据集不需要复杂的模板匹配，直接按单词加载即可。
    """
    audio_dict = {}
    print(f"正在预加载音频模板 (目标采样率：{TARGET_SR_AUDIO})...")

    for word in WORDS_SEQ:
        wav_path = os.path.join(ORIG_WAV_DIR, f"{word}.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"错误：找不到音频文件 {wav_path}")

        wav_tensor, sr = torchaudio.load(wav_path)

        # 重采样
        if sr != TARGET_SR_AUDIO:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, TARGET_SR_AUDIO)

        # 转为单声道
        if wav_tensor.shape[0] > 1:
            wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

        # 裁剪或填充到固定长度 (DURATION * TARGET_SR_AUDIO)
        target_samples = int(DURATION * TARGET_SR_AUDIO)
        if wav_tensor.shape[1] < target_samples:
            pad_tensor = torch.zeros(1, target_samples - wav_tensor.shape[1])
            wav_tensor = torch.cat([wav_tensor, pad_tensor], dim=1)
        else:
            wav_tensor = wav_tensor[:, :target_samples]

        # 转为 numpy 以便后续处理
        audio_dict[word] = wav_tensor.squeeze().numpy()

    return audio_dict


def read_public_dataset(subjects, sessions):
    """
    读取公共数据集 (Semantics-EEG-Perception-and-Imagination)
    """
    all_eeg_data = []  # 存储 (samples, channels, time)
    all_labels = []  # 存储 label indices
    audio_templates = load_audio_templates()  # 预加载音频，用于匹配样本数量和获取语音数据

    base_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_{TASK}\\'

    for subject in subjects:
        for session in sessions:
            datapoint = f'{subject}_{session}_epo.fif'
            full_path = os.path.join(base_path, datapoint)

            try:
                print(f"正在读取: {full_path}")
                epochs = read_epochs(full_path)
                epochs.crop(tmin=0, tmax=DURATION)  # 裁剪到指定时长

                # 更新事件ID映射
                event_id_mapping = {k: v for k, v in WORD_TO_LABEL.items()}

                # 这里简化处理：假设 epochs.event_id 中的 key 包含我们需要的单词
                # 例如: 'perc_flower_s' -> 我们需要提取 'flower'
                valid_data = []
                valid_labels = []

                for event_name, event_idx in epochs.event_id.items():
                    # 解析事件名，提取单词 (假设格式为 perc_word_tag)
                    parts = event_name.split('_')
                    if len(parts) >= 3 and parts[1] in WORD_TO_LABEL:
                        word = parts[1]
                        label = WORD_TO_LABEL[word]

                        # 获取该类别的所有 trial 索引
                        trial_idx = np.where(epochs.events[:, 2] == event_idx)[0]

                        # 获取数据 (形状: n_trials, n_channels, n_times)
                        data = epochs.get_data(copy=False)[trial_idx]

                        valid_data.append(data)
                        valid_labels.extend([label] * len(data))

                if valid_data:
                    subject_data = np.concatenate(valid_data, axis=0)
                    all_eeg_data.append(subject_data)
                    all_labels.extend(valid_labels)

            except Exception as e:
                print(f"读取 {datapoint} 出错：{e}")
                continue

    if not all_eeg_data:
        raise ValueError("没有读取到任何有效的 EEG 数据，请检查路径和文件格式。")

    # 合并所有被试的数据
    eeg_array = np.concatenate(all_eeg_data, axis=0)
    labels_array = np.array(all_labels)

    print(f"读取完成，总样本数: {eeg_array.shape[0]}")

    # 准备返回的 segments 格式：[(eeg_seg, audio_seg, label), ...]
    # 注意：这里假设每个 EEG trial 对应一个固定的音频模板（按类别）
    segments = []
    for i in range(len(eeg_array)):
        word = WORDS_SEQ[labels_array[i]]
        eeg_seg = eeg_array[i]  # 形状已经是 (channels, time)
        audio_seg = audio_templates[word]  # 获取对应的音频
        segments.append((eeg_seg, audio_seg, labels_array[i]))

    return segments


def save_segment(data_item, local_idx, split_name):
    """
    保存单个样本。
    注意：公共数据集形状是 (channels, timestamps)，不需要转置。
    """
    prefix_num = f"{local_idx:04d}"
    subj_id = "sub-public"  # 标记为公共数据集
    name_eeg = f"{subj_id}_ts-{prefix_num}.csv"
    name_voice = f"{subj_id}_voice_ts-{prefix_num}.csv"
    name_mel = f"{subj_id}_mel_ts-{prefix_num}.csv"
    name_label = f"{subj_id}_ts-{prefix_num}_label.csv"

    base_path = os.path.join(OUTPUT_ROOT, split_name)
    eeg_raw, audio_raw, label = data_item

    # 1. 保存 EEG (直接使用 StandardScaler，形状不变)
    scaler = StandardScaler()
    # eeg_raw 形状: (24, 2048)
    eeg_norm = scaler.fit_transform(eeg_raw)
    df_eeg = pd.DataFrame(eeg_norm)
    df_eeg.to_csv(os.path.join(base_path, 'SpokenEEG', name_eeg), index=False, header=False)

    # 2. 保存 Voice
    audio_scaled = audio_raw * MAX_WAV_VALUE
    pd.DataFrame(audio_scaled.reshape(-1, 1)).to_csv(
        os.path.join(base_path, 'Voice', name_voice),
        index=False, header=False
    )

    # 3. 保存 Mel (使用 HiFi-GAN 配置提取)
    try:
        wav_tensor = torch.from_numpy(audio_raw).float().unsqueeze(0).to(device)
        with torch.no_grad():
            mel_spec = mel_transform(wav_tensor)
            log_mel_spec = torch.log(mel_spec + 1e-5)
            log_mel_np = log_mel_spec.squeeze(0).cpu().numpy()
        df_mel = pd.DataFrame(log_mel_np)
        df_mel.to_csv(
            os.path.join(base_path, 'Voice_mel', name_mel),
            index=False, header=False
        )
    except Exception as e:
        print(f"提取 Mel 频谱时出错：{e}")
        df_mel = pd.DataFrame(np.zeros((hifigan_config['num_mels'], 10)))
        df_mel.to_csv(os.path.join(base_path, 'Voice_mel', name_mel), index=False, header=False)

    # 4. 保存 Label (One-hot)
    one_hot = np.zeros(NUM_CLASSES, dtype=int)
    if 0 <= label < NUM_CLASSES:
        one_hot[label] = 1
    df_y = pd.DataFrame([one_hot])
    df_y.to_csv(
        os.path.join(base_path, 'Y', name_label),
        index=False, header=False, float_format='%.0f'
    )


def main():
    # 1. 初始化组件
    try:
        setup_hifigan()
    except Exception as e:
        print(f"严重错误：初始化 HiFi-GAN 失败 - {e}")
        return

    create_directories()

    # 2. 读取公共数据集
    try:
        all_segments = read_public_dataset(SUBJECTS, SESSIONS)
        total_count = len(all_segments)
        if total_count == 0:
            print("错误：没有生成任何数据。")
            return
        print(f"\n总共生成 {total_count} 个样本片段。")
    except Exception as e:
        print(f"读取公共数据集失败: {e}")
        return

    # 3. 简单划分数据集 (每10个取1个做测试/验证)
    test_indices = set([i for i in range(0, total_count, 10)])
    train_data = [all_segments[i] for i in range(total_count) if i not in test_indices]
    test_data = [all_segments[i] for i in test_indices]
    val_data = test_data  # 验证集与测试集相同，或者也可以单独划分

    print(f"设置完成 -> 训练集：{len(train_data)}, 测试集：{len(test_data)}, 验证集：{len(val_data)}")

    # 4. 保存数据
    print("\n保存训练集...")
    for idx, item in enumerate(train_data, 1):
        save_segment(item, idx, 'train')

    print("保存测试集...")
    for idx, item in enumerate(test_data, 1):
        save_segment(item, idx, 'test')

    print("保存验证集...")
    for idx, item in enumerate(val_data, 1):
        save_segment(item, idx, 'val')

    print(f"\n✅ 处理完成！数据已保存至：{os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()