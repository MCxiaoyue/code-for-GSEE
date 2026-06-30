from pymcd.mcd import Calculate_MCD
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import numpy as np
import os
import librosa
import fnmatch  # 用于文件名匹配

# ================= 配置区域 =================
# 假设所有文件都在同一个文件夹下，如果分开请改为两个路径
data_folder = "./TrainResult_models1_1/sub1_SpokenEEG_EVR_Hybrid/savevoice"  # 包含所有 .wav 文件的文件夹

# 定义文件名模式
# Target 文件模式: 例如 001_Target.wav
target_suffix = "_Target.wav"
# Recon 文件模式: 例如 001_Recon_SP_*-pred_.wav
# 我们只需要前缀和后半部分的固定特征
recon_prefix_part = "_Recon_SP_"
recon_suffix_part = "-pred_.wav"
# ===========================================

# 创建 MCD 工具实例
mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")

# 存储结果
results = []

# 获取所有 _Target.wav 文件
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"文件夹不存在: {data_folder}")

all_files = os.listdir(data_folder)
target_files = [f for f in all_files if f.endswith(target_suffix)]

print(f"Found {len(target_files)} target files. Starting evaluation...\n")


# 定义频谱计算函数
def compute_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec


for target_file in target_files:
    # 1. 提取 ID (基础名称)
    # 例如: '001_Target.wav' -> '001'
    base_id = target_file.replace(target_suffix, "")

    # 2. 寻找对应的 Recon 文件
    # 构造搜索模式: 001_Recon_SP_*-pred_.wav
    # 使用 fnmatch 过滤，或者遍历查找
    matched_recon_file = None

    # 构建前缀用于快速筛选
    search_prefix = f"{base_id}{recon_prefix_part}"

    for f in all_files:
        if f.startswith(search_prefix) and f.endswith(recon_suffix_part):
            matched_recon_file = f
            break

    if not matched_recon_file:
        print(f"Skipping {target_file}: No matching recon file found (Pattern: {base_id}_Recon_SP_*-pred_.wav)")
        continue

    target_path = os.path.join(data_folder, target_file)
    recon_path = os.path.join(data_folder, matched_recon_file)

    try:
        # --- MCD 计算 ---
        # pymcd 的 calculate_mcd 通常直接接受文件路径
        mcd_value = mcd_toolbox.calculate_mcd(target_path, recon_path)

        # --- 加载音频 ---
        y_target, sr_target = librosa.load(target_path, sr=None)
        y_recon, sr_recon = librosa.load(recon_path, sr=None)

        if sr_target != sr_recon:
            # 如果采样率不同，librosa.load 时可以强制指定 sr，或者报错
            # 这里选择重采样到一致 (以 target 为准)
            print(f"Warning: Sample rates differ ({sr_target} vs {sr_recon}). Resampling recon to {sr_target}.")
            y_recon, sr_recon = librosa.load(recon_path, sr=sr_target)

        # 裁剪为相同长度 (防止长度不一致导致后续计算报错)
        min_len = min(len(y_target), len(y_recon))
        if min_len == 0:
            raise ValueError("Audio length is zero.")

        y_target = y_target[:min_len]
        y_recon = y_recon[:min_len]

        # --- 梅尔频谱计算 ---
        spec_target = compute_spectrogram(y_target, sr_target)
        spec_recon = compute_spectrogram(y_recon, sr_target)

        # 裁剪频谱图时间维度以完全匹配 (虽然音频裁剪了，但librosa计算spectrogram可能会有细微帧数差异)
        min_time = min(spec_target.shape[1], spec_recon.shape[1])
        spec_target = spec_target[:, :min_time]
        spec_recon = spec_recon[:, :min_time]

        # --- SSIM ---
        # data_range 需要根据数据的动态范围计算
        data_range = max(spec_target.max(), spec_recon.max()) - min(spec_target.min(), spec_recon.min())
        if data_range == 0:
            data_range = 1.0  # 避免除以零或警告
        ssim_value = ssim(spec_target, spec_recon, data_range=data_range)

        # --- PCC: 按帧平均后计算时序相关性 ---
        # 原始代码逻辑：对每一帧的所有频率点求平均，得到一个时间序列，然后计算两个时间序列的相关性
        pcc_value = pearsonr(spec_target.mean(axis=0), spec_recon.mean(axis=0))[0]

        # 输出当前结果
        print(f"Evaluation for ID '{base_id}':")
        print(f"  Target: {target_file}")
        print(f"  Recon : {matched_recon_file}")
        print(f"  MCD   : {mcd_value:.4f}")
        print(f"  SSIM  : {ssim_value:.4f}")
        print(f"  PCC   : {pcc_value:.4f}")
        print('------------------------------')

        # 保存结果
        results.append({
            'id': base_id,
            'target_file': target_file,
            'recon_file': matched_recon_file,
            'MCD': mcd_value,
            'SSIM': ssim_value,
            'PCC': pcc_value
        })

    except Exception as e:
        print(f"Error processing {base_id}: {e}")
        import traceback

        traceback.print_exc()

# --- 汇总统计 ---
if results:
    mcd_list = [r['MCD'] for r in results]
    ssim_list = [r['SSIM'] for r in results]
    pcc_list = [r['PCC'] for r in results]

    print("\n📊 Overall Evaluation Results:")
    print(f"Total valid pairs: {len(results)}")
    print(f"MCD   : Average = {np.mean(mcd_list):.4f}, Std = {np.std(mcd_list):.4f}")
    print(f"SSIM  : Average = {np.mean(ssim_list):.4f}, Std = {np.std(ssim_list):.4f}")
    print(f"PCC   : Average = {np.mean(pcc_list):.4f}, Std = {np.std(pcc_list):.4f}")

    # 可选：保存结果到 CSV
    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv("evaluation_results.csv", index=False)
    # print("Results saved to evaluation_results.csv")
else:
    print("No valid file pairs were found or processed. Please check your file naming patterns.")