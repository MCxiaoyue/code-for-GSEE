from pymcd.mcd import Calculate_MCD
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import numpy as np
import os
import librosa

# 设置文件夹路径
pre_folder = "./eeg_data_=_testset"   # 包含 *_pre.wav 的文件夹
re_folder = "./testset"              # 包含 *_re.wav 的文件夹

# 创建 MCD 工具实例
mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")

# 存储结果
results = []

# 获取所有 _pre.wav 文件
pre_files = [f for f in os.listdir(pre_folder) if f.endswith("_pre.wav")]

print("Starting evaluation...\n")

# 定义频谱计算函数
def compute_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec

for pre_file in pre_files:
    # 提取基础名称，例如：'abc_pre.wav' -> 'abc'
    base_name = pre_file[:-13]  # 去掉 '_pred_pre.wav'
    re_file = base_name + "_re.wav"

    pre_path = os.path.join(pre_folder, pre_file)
    re_path = os.path.join(re_folder, re_file)

    # 检查 re 文件是否存在
    if not os.path.exists(re_path):
        print(f"Skipping {pre_file}: Corresponding _re.wav file not found.")
        continue

    try:
        # --- MCD 计算 ---
        mcd_value = mcd_toolbox.calculate_mcd(pre_path, re_path)

        # --- 加载音频 ---
        y_pre, sr_pre = librosa.load(pre_path, sr=None)
        y_re, sr_re = librosa.load(re_path, sr=None)

        if sr_pre != sr_re:
            raise ValueError("Sample rates do not match.")

        # 裁剪为相同长度
        min_len = min(len(y_pre), len(y_re))
        y_pre = y_pre[:min_len]
        y_re = y_re[:min_len]

        # --- 梅尔频谱计算 ---
        spec_pre = compute_spectrogram(y_pre, sr_pre)
        spec_re = compute_spectrogram(y_re, sr_pre)

        # 裁剪频谱图时间维度以匹配
        min_time = min(spec_pre.shape[1], spec_re.shape[1])
        spec_pre = spec_pre[:, :min_time]
        spec_re = spec_re[:, :min_time]

        # --- SSIM ---
        data_range = max(spec_pre.max(), spec_re.max()) - min(spec_pre.min(), spec_re.min())
        ssim_value = ssim(spec_pre, spec_re, data_range=data_range)

        # --- PCC: 按帧平均后计算时序相关性 ---
        pcc_value = pearsonr(spec_pre.mean(axis=0), spec_re.mean(axis=0))[0]

        # 输出当前结果
        print(f"Evaluation for '{base_name}':")
        print(f"  MCD:  {mcd_value:.4f}")
        print(f"  SSIM: {ssim_value:.4f}")
        print(f"  PCC:  {pcc_value:.4f}")
        print('------------------------------')

        # 保存结果
        results.append({
            'name': base_name,
            'MCD': mcd_value,
            'SSIM': ssim_value,
            'PCC': pcc_value
        })

    except Exception as e:
        print(f"Error processing {base_name}: {e}")

# --- 汇总统计 ---
if results:
    mcd_list = [r['MCD'] for r in results]
    ssim_list = [r['SSIM'] for r in results]
    pcc_list = [r['PCC'] for r in results]

    print("\n📊 Overall Evaluation Results:")
    print(f"Total valid pairs: {len(results)}")
    print(f"MCD:  Average = {np.mean(mcd_list):.4f}, Std = {np.std(mcd_list):.4f}")
    print(f"SSIM: Average = {np.mean(ssim_list):.4f}, Std = {np.std(ssim_list):.4f}")
    print(f"PCC:  Average = {np.mean(pcc_list):.4f}, Std = {np.std(pcc_list):.4f}")
else:
    print("No valid file pairs were found or processed.")