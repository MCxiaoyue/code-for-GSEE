import os
import torch
import torchaudio
import json
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 原始音频文件夹
INPUT_DIR = "./orign_wav_public"
# 输出文件夹 (如果设为 None，则直接覆盖原文件；建议设为新文件夹以备份)
OUTPUT_DIR = "./orign_wav_public_resampled"

# HiFi-GAN 配置文件路径 (用于获取目标采样率)
HIFIGAN_CONFIG_PATH = './pretrained_model/UNIVERSAL_V1/config.json'


# ===========================================

def get_target_sr(config_path):
    """从 config.json 读取目标采样率"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到：{config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 常见的键名可能是 sampling_rate 或 sample_rate
    target_sr = config.get('sampling_rate') or config.get('sample_rate')

    if not target_sr:
        raise ValueError("配置文件中未找到 sampling_rate 或 sample_rate 字段")

    return int(target_sr)


def resample_file(input_path, output_path, target_sr):
    """单个文件的重采样逻辑"""
    try:
        # 1. 加载音频
        waveform, orig_sr = torchaudio.load(input_path)

        # 如果已经是目标采样率，直接复制（可选）
        if orig_sr == target_sr:
            print(f"跳过 (采样率已匹配): {os.path.basename(input_path)}")
            # 如果输出目录不同，仍需保存
            if output_path != input_path:
                torchaudio.save(output_path, waveform, orig_sr)
            return True

        # 2. 执行重采样
        transform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled_waveform = transform(waveform)

        # 3. 保存文件
        # 注意：torchaudio.save 会自动根据扩展名决定格式，这里默认 wav
        torchaudio.save(output_path, resampled_waveform, target_sr)
        return True

    except Exception as e:
        print(f"❌ 处理失败 {os.path.basename(input_path)}: {e}")
        return False


def main():
    # 1. 获取目标采样率
    print(f"正在读取配置：{HIFIGAN_CONFIG_PATH} ...")
    try:
        target_sr = get_target_sr(HIFIGAN_CONFIG_PATH)
        print(f"✅ 目标采样率确定为：{target_sr} Hz")
    except Exception as e:
        print(f"❌ 读取配置失败：{e}")
        print("💡 提示：请检查 config.json 路径是否正确，或手动设置 target_sr 变量。")
        return

    # 2. 准备输入输出路径
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"❌ 输入文件夹不存在：{INPUT_DIR}")
        return

    # 确定输出模式
    overwrite = False
    if OUTPUT_DIR is None:
        overwrite = True
        out_path = input_path
        print("⚠️  模式：将直接覆盖原文件！")
    else:
        out_path = Path(OUTPUT_DIR)
        out_path.mkdir(exist_ok=True)
        print(f"📂 模式：保存到 {out_path}")

    # 3. 获取所有 wav 文件
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
    # 去重 (防止大小写敏感系统重复)
    wav_files = list(set(wav_files))

    if not wav_files:
        print("❌ 未在文件夹中找到任何 .wav 文件。")
        return

    print(f"🔍 找到 {len(wav_files)} 个音频文件，开始处理...\n")

    # 4. 批量处理
    success_count = 0
    fail_count = 0

    for file_path in tqdm(wav_files, desc="Resampling"):
        if overwrite:
            dest_path = file_path
        else:
            dest_path = out_path / file_path.name

        if resample_file(str(file_path), str(dest_path), target_sr):
            success_count += 1
        else:
            fail_count += 1

    # 5. 总结
    print("\n" + "=" * 30)
    print(f"🎉 处理完成！")
    print(f"   成功：{success_count} 个")
    print(f"   失败：{fail_count} 个")
    if not overwrite:
        print(f"   输出位置：{os.path.abspath(out_path)}")
    print("=" * 30)


if __name__ == "__main__":
    # 自动检测 GPU 加速 (torchaudio 的 Resample 在某些版本支持 CUDA，但 CPU 通常也足够快)
    # 如果需要强制 CPU 以防显存溢出，可以在 resample_file 中加 .cpu()
    main()