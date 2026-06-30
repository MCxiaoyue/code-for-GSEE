import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.transforms as T
import json

# 尝试导入你的本地模型文件，如果路径不同请修改
# 假设目录结构为: ./models/models_HiFi.py 和 ./modules.py
try:
    from models.models_HiFi import Generator as model_HiFi
    from modules import AttrDict
except ImportError as e:
    print(f"警告：无法导入 HiFi-GAN 模型模块 ({e})。请检查 models/ 和 modules.py 是否存在。")
    print("如果不使用声码器验证，仅提取 Mel，代码仍可运行，但需确保 config.json 存在。")
    model_HiFi = None
    AttrDict = None

# =================配置区域=================
ORIG_EEG_DIR = "./orign_eeg_data"
ORIG_WAV_DIR = "./orign_wav_resampled"
OUTPUT_ROOT = "./processed_dataset_1_beifen/sub1"

# HiFi-GAN 配置文件路径 (请根据实际情况修改)
HIFIGAN_CONFIG_PATH = './pretrained_model/UNIVERSAL_V1/config.json'
HIFIGAN_VOCODER_PATH = './pretrained_model/UNIVERSAL_V1/g_02500000'

ALL_FILES = [
    'vDLY-001_1.txt', 'vLCL-001_1.txt', 'vLHJ-001_1.txt', 'vLHJ-002_1.txt',
    'vLCL-004_1.txt', 'vYHC02_1.txt', 'vLCL-01_1.txt', 'vLCL-02_1.txt',
    'vCLB-001_1.txt', 'vYHC-001_1.txt', 'vLCL-003_1.txt'
]

WORDS_SEQ = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]

WORD_TO_LABEL = {
    "my": 8,
    "dad": 0,
    "is": 1,
    "a": 2,
    "policeman": 3,
    "he": 4,
    "will": 5,
    "always": 6,
    "become": 7,
    "hero": 9
}

SAMPLING_RATE_EEG = 128
TARGET_SR_AUDIO = 22050  # 注意：HiFi-GAN 通常使用 22050 或 24000，需与 config.json 一致
WORD_DURATION = 1.5
REST_WITHIN_WORD = 5.0
REST_BETWEEN_WORDS = 10.0
REPETITIONS = 5

N_MELS = 80  # 这个值将被 config.json 中的 num_mels 覆盖，以保持兼容性
HOP_LENGTH = 256
N_FFT = 1024

NUM_CLASSES = 10
MAX_WAV_VALUE = 32768

# 全局变量用于存储加载好的模型和变换器
hifigan_config = None
mel_transform = None
vocoder_model = None
device = None


# =========================================

def setup_hifigan():
    """初始化 HiFi-GAN 配置和 Mel 变换器"""
    global hifigan_config, mel_transform, vocoder_model, device, TARGET_SR_AUDIO

    if not os.path.exists(HIFIGAN_CONFIG_PATH):
        raise FileNotFoundError(f"未找到 HiFi-GAN 配置文件：{HIFIGAN_CONFIG_PATH}")

    print("正在加载 HiFi-GAN 配置...")
    with open(HIFIGAN_CONFIG_PATH, 'r') as f:
        h_dict = json.load(f)

    if AttrDict:
        hifigan_config = AttrDict(h_dict)
    else:
        # 如果没有导入成功，使用普通字典兼容
        hifigan_config = h_dict

    # 更新全局采样率以匹配模型配置
    TARGET_SR_AUDIO = hifigan_config.sampling_rate
    print(f"HiFi-GAN 采样率设定为：{TARGET_SR_AUDIO}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 定义 Mel 变换器 (参数严格匹配 config.json)
    # 注意：torchaudio 的默认行为可能与 librosa 略有不同，这里强制指定 slaney
    mel_transform = T.MelSpectrogram(
        sample_rate=hifigan_config.sampling_rate,
        n_fft=hifigan_config.n_fft,
        win_length=hifigan_config.win_size,
        hop_length=hifigan_config.hop_size,
        f_min=hifigan_config.fmin,
        f_max=hifigan_config.fmax,
        n_mels=hifigan_config.num_mels,
        power=1,
        normalized=False,
        norm='slaney',
        mel_scale='slaney'
    ).to(device)
    mel_transform.eval()

    # (可选) 加载声码器用于验证，如果不需要还原语音可注释掉以节省显存
    if model_HiFi and os.path.exists(HIFIGAN_VOCODER_PATH):
        print("正在加载 HiFi-GAN 声码器 (用于验证)...")
        vocoder_model = model_HiFi(hifigan_config).to(device)
        state_dict = torch.load(HIFIGAN_VOCODER_PATH, map_location='cpu')
        # 兼容不同版本的 state_dict 键名
        if 'generator' in state_dict:
            vocoder_model.load_state_dict(state_dict['generator'])
        else:
            vocoder_model.load_state_dict(state_dict)
        vocoder_model.eval()
        print("声码器加载完成。")
    else:
        print("未加载声码器模型（可能缺少文件或模块），将仅提取 Mel 频谱。")
        vocoder_model = None


def create_directories():
    splits = ['train', 'test', 'val']
    subfolders = ['SpokenEEG', 'Voice', 'Voice_mel', 'Y']
    for split in splits:
        for sub in subfolders:
            path = os.path.join(OUTPUT_ROOT, split, sub)
            os.makedirs(path, exist_ok=True)
    print("目录结构创建完成。")


def load_audio_templates(word_list, wav_dir, target_sr):
    audio_dict = {}
    print(f"正在预加载音频模板 (目标采样率：{target_sr})...")
    for word in set(word_list):
        wav_path = os.path.join(wav_dir, f"{word}.wav")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"错误：找不到音频文件 {wav_path}")
        # torchaudio 加载通常更快，且直接返回 Tensor
        wav_tensor, sr = torchaudio.load(wav_path)

        # 重采样如果必要
        if sr != target_sr:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, target_sr)

        # 转为 mono (如果 stereo)
        if wav_tensor.shape[0] > 1:
            wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

        # 转为 numpy 用于后续切片操作 (保持与原有逻辑兼容，或者全程用 tensor)
        # 这里为了配合 extract_segments 中的 numpy 操作，先转回 numpy
        audio_dict[word] = wav_tensor.squeeze().numpy()
    return audio_dict


def extract_segments_with_template(eeg_data, audio_templates):
    segments = []
    samples_per_word = int(WORD_DURATION * SAMPLING_RATE_EEG)
    samples_rest_within = int(REST_WITHIN_WORD * SAMPLING_RATE_EEG)
    samples_rest_between = int(REST_BETWEEN_WORDS * SAMPLING_RATE_EEG)

    # 音频样本数基于新的 TARGET_SR_AUDIO (从 config 读取)
    audio_samples_per_word = int(WORD_DURATION * TARGET_SR_AUDIO)

    current_eeg_idx = 0

    for word in WORDS_SEQ:
        if word not in audio_templates:
            raise ValueError(f"音频模板中缺少单词：{word}")
        full_audio = audio_templates[word]  # numpy array

        for rep in range(REPETITIONS):
            eeg_start = current_eeg_idx
            eeg_end = eeg_start + samples_per_word

            if eeg_end > len(eeg_data):
                break

            eeg_seg = eeg_data[eeg_start:eeg_end, :]

            # 处理音频长度
            if len(full_audio) < audio_samples_per_word:
                audio_seg = np.pad(full_audio, (0, audio_samples_per_word - len(full_audio)), mode='constant')
            else:
                audio_seg = full_audio[:audio_samples_per_word]

            segments.append((eeg_seg, audio_seg, WORD_TO_LABEL[word]))

            if rep < REPETITIONS - 1:
                current_eeg_idx += samples_per_word + samples_rest_within
            else:
                current_eeg_idx += samples_per_word + samples_rest_between

    return segments


def process_all_files_to_memory(file_list, audio_templates):
    all_segments = []
    print("\n正在读取并切分 EEG 数据...")
    for file_name in tqdm(file_list, desc="Processing EEG Files"):
        eeg_path = os.path.join(ORIG_EEG_DIR, file_name)
        if not os.path.exists(eeg_path):
            continue

        eeg_raw = []
        try:
            with open(eeg_path, 'r') as f:
                lines = f.readlines()
                # 保持原有的切片逻辑
                target_lines = lines[1282:52844]
                for line in target_lines:
                    parts = line.strip().split('\t')
                    if len(parts) < 25:
                        continue
                    row = [float(x) for x in parts[1:25]]
                    eeg_raw.append(row)
        except Exception as e:
            print(f"读取 {file_name} 出错：{e}")
            continue

        eeg_np = np.array(eeg_raw)
        segments = extract_segments_with_template(eeg_np, audio_templates)
        all_segments.extend(segments)

    return all_segments


def save_segment(data_item, local_idx, split_name):
    prefix_num = f"{local_idx:04d}"
    subj_id = "sub-01"

    name_eeg = f"{subj_id}_ts-{prefix_num}.csv"
    name_voice = f"{subj_id}_voice_ts-{prefix_num}.csv"
    name_mel = f"{subj_id}_mel_ts-{prefix_num}.csv"
    name_label = f"{subj_id}_ts-{prefix_num}_label.csv"

    base_path = os.path.join(OUTPUT_ROOT, split_name)

    eeg_raw, audio_raw, label = data_item

    # print(eeg_raw.shape)
    # print('=================================')

    # 1. 保存 EEG
    scaler = StandardScaler()
    eeg_norm = scaler.fit_transform(eeg_raw.T).T
    df_eeg = pd.DataFrame(eeg_norm.T)
    # df_eeg = pd.DataFrame(eeg_norm)
    df_eeg.to_csv(
        os.path.join(base_path, 'SpokenEEG', name_eeg), index=False, header=False)
    # df_eeg = pd.DataFrame(eeg_raw)
    # df_eeg.to_csv(
    #     os.path.join(base_path, 'SpokenEEG', name_eeg), index=False, header=False)

    # 2. 保存 Voice (原始音频缩放到 int16 范围)
    audio_scaled = audio_raw * MAX_WAV_VALUE
    pd.DataFrame(audio_scaled.reshape(-1, 1)).to_csv(
        os.path.join(base_path, 'Voice', name_voice), index=False, header=False)

    # 3. 保存 Mel (使用 HiFi-GAN 配置提取) - [已添加调试诊断]
    try:
        # 将 numpy 音频转为 torch tensor [1, Time]
        # 检查 audio_raw 的范围
        audio_min = audio_raw.min()
        audio_max = audio_raw.max()

        # # 【诊断 1】检查输入音频幅度
        # if local_idx <= 5:  # 只打印前5个样本
        #     print(f"\n[Debug Sample {local_idx}] Input Audio Range: [{audio_min:.4f}, {audio_max:.4f}]")
        #     if audio_max < 0.1:
        #         print("⚠️ 警告：输入音频幅度过小！可能导致 Mel 能量过低。")

        wav_tensor = torch.from_numpy(audio_raw).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. 提取 Mel 频谱 (线性刻度)
            mel_spec = mel_transform(wav_tensor)

            # 2. 转换为 Log-Mel (dB)
            log_mel_spec = torch.log(mel_spec + 1e-5)

        # 移除 batch 维度
        log_mel_np = log_mel_spec.squeeze(0).cpu().numpy()

        df_mel = pd.DataFrame(log_mel_np)
        df_mel.to_csv(
            os.path.join(base_path, 'Voice_mel', name_mel),
            index=False,
            header=False
        )

    except Exception as e:
        print(f"提取 Mel 频谱时出错：{e}")
        import traceback
        traceback.print_exc()
        df_mel = pd.DataFrame(np.zeros((hifigan_config.num_mels, 10)))
        df_mel.to_csv(os.path.join(base_path, 'Voice_mel', name_mel), index=False, header=False)

    # 4. 保存 Label
    one_hot = np.zeros(NUM_CLASSES, dtype=int)
    if 0 <= label < NUM_CLASSES:
        one_hot[label] = 1
    df_y = pd.DataFrame([one_hot])
    df_y.to_csv(
        os.path.join(base_path, 'Y', name_label), index=False, header=False, float_format='%.0f')


def main():
    # 1. 初始化 HiFi-GAN 组件
    try:
        setup_hifigan()
    except Exception as e:
        print(f"严重错误：初始化 HiFi-GAN 失败 - {e}")
        return

    create_directories()

    try:
        # 注意：load_audio_templates 现在会使用更新后的 TARGET_SR_AUDIO
        audio_templates = load_audio_templates(WORDS_SEQ, ORIG_WAV_DIR, TARGET_SR_AUDIO)
    except FileNotFoundError as e:
        print(e)
        return

    all_segments = process_all_files_to_memory(ALL_FILES, audio_templates)
    total_count = len(all_segments)

    if total_count == 0:
        print("错误：没有生成任何数据。")
        return

    print(f"\n总共生成 {total_count} 个样本片段。")
    print(
        f"Mel 频谱维度应为：{hifigan_config.num_mels} x ~{int(WORD_DURATION * TARGET_SR_AUDIO / hifigan_config.hop_size)}")

    # train_data = all_segments
    # test_data = all_segments
    # val_data = all_segments

    # 简单的划分：每10个取1个做测试
    test_indices = [i for i in range(0, total_count, 10)]
    test_indices_set = set(test_indices)

    test_data = [all_segments[i] for i in test_indices]
    train_data = [all_segments[i] for i in range(total_count) if i not in test_indices_set]
    val_data = test_data

    print(f"设置完成 -> 训练集：{len(train_data)}, 测试集：{len(test_data)}, 验证集：{len(val_data)}")

    # 保存训练集
    print("\n保存训练集...")
    for idx, item in enumerate(train_data, 1):
        save_segment(item, idx, 'train')

    # 保存测试集
    print("保存测试集...")
    for idx, item in enumerate(test_data, 1):
        save_segment(item, idx, 'test')

    # 保存验证集
    print("保存验证集...")
    for idx, item in enumerate(val_data, 1):
        save_segment(item, idx, 'val')

    print(f"\n✅ 处理完成！数据已保存至：{os.path.abspath(OUTPUT_ROOT)}")
    print("📊 最终数据形状确认:")
    print(f"   - EEG CSV: 行=时间 (~192), 列=通道 (24)")
    print(f"   - Mel CSV: 行={hifigan_config.num_mels}, 列=时间 (取决于 hop_size)")
    print(f"   - Label CSV: 1行 x {NUM_CLASSES}列")


if __name__ == "__main__":
    main()