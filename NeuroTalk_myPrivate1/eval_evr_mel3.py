import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import time
import torchaudio
import wavio
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate
from tqdm import tqdm

# ==========================================
# 1. 引入必要的模块
# ==========================================
try:
    from models.models1_2 import Discriminator, HybridGenerator
    from models.models_HiFi import Generator as model_HiFi
except ImportError as e:
    print(f"❌ Error: Cannot import models. Please check the models path. {e}")
    exit()

from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
from NeuroTalkDataset import myDataset


def load_checkpoint(model, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded epoch: {checkpoint.get('epoch', 'Unknown')}")
        else:
            model.load_state_dict(checkpoint)
        return True
    else:
        print(f"❌ No checkpoint found at '{checkpoint_path}'")
        return False


def generat(args, test_loader, models):
    """
    执行测试/推理流程
    """
    model_g, model_d, vocoder, model_STT, decoder_STT = models
    model_g.eval()
    model_d.eval()
    vocoder.eval()
    model_STT.eval()

    criterion_recon = RMSELoss().to(args.device)
    criterion_cl = nn.CrossEntropyLoss().to(args.device)
    CER = CharErrorRate().to(args.device)

    test_losses = []
    save_idx = 0 # 初始化保存索引

    print(f"🚀 Starting Inference on {len(test_loader.dataset)} samples...")

    # --- 确保保存目录存在 ---
    os.makedirs(args.savevoice, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
            input = input.to(args.device)
            target = target.to(args.device)
            target_cl = target_cl.to(args.device)
            # 注意：旧脚本中 voice 被 squeeze 且没有转移到 device，这里保持旧逻辑特征
            voice_np = torch.squeeze(voice, dim=-1).cpu().numpy()
            labels = torch.argmax(target_cl, dim=1)

            # --- 1. 前向传播 & 对齐 ---
            output = model_g(input)
            mel_out = DTW_align(output, target)

            # --- 2. 反归一化 (旧脚本逻辑：对 Target 和 Output 都进行反归一化) ---
            # 获取归一化参数
            mean = data_info[0].to(args.device)
            std = data_info[1].to(args.device)
            target_denorm = data_denorm(target, mean, std)
            mel_out_denorm = data_denorm(mel_out, mean, std)

            # --- 3. Vocoder 转换 ---
            # 旧脚本中 batch_size 参数传的是 1
            wav_target = mel2wav_vocoder(target_denorm, vocoder, 1)
            wav_recon = mel2wav_vocoder(mel_out_denorm, vocoder, 1)

            # --- 4. 形状调整与重采样 ---
            # Reshape
            wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))
            wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))

            # 重采样到 STT 采样率
            wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
            wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)

            # 转换为 numpy 以便保存
            wav_target_np = wav_target.cpu().detach().numpy()
            wav_recon_np = wav_recon.cpu().detach().numpy()

            # --- 5. 保存音频文件 (核心新增部分) ---
            for batch_idx in range(len(input)):
                # 获取标签文本
                try:
                    # 防止索引错误
                    gt_idx = labels[batch_idx].item()
                    str_tar = args.word_label[gt_idx].replace("|", ",").replace(" ", ",")
                except:
                    str_tar = "Unknown"

                # --- 保存 重建音频 (Recon) ---
                # (这里保留了新脚本的逻辑，你可以根据需要调整命名)
                if args.task[0] == 'I':
                    title_recon = f"Recon_IM_{str_tar}"
                else:
                    title_recon = f"Recon_SP_{str_tar}"
                wavio.write(os.path.join(args.savevoice, f"{save_idx:03d}_{title_recon}.wav"),
                           wav_recon_np[batch_idx], args.sample_rate_STT, sampwidth=1)

                # --- 保存 目标音频 (Target) ---
                # 对应旧脚本中的 "Target"
                title_target = "Target"
                wavio.write(os.path.join(args.savevoice, f"{save_idx:03d}_{title_target}.wav"),
                           wav_target_np[batch_idx], args.sample_rate_STT, sampwidth=1)

                # --- 保存 原始音频 (Original) ---
                # 对应旧脚本中的 "Original"
                # 旧脚本特征：硬编码采样率为 22050，sampwidth=1
                title_origin = "Original"
                wavio.write(os.path.join(args.savevoice, f"{save_idx:03d}_{title_origin}.wav"),
                           voice_np[batch_idx], 22050, sampwidth=1) # 保留旧脚本硬编码特征

                save_idx += 1

            # 记录损失
            loss_recon = criterion_recon(mel_out, target)
            test_losses.append(loss_recon.item())
            pbar.set_postfix({'Recon Loss': f'{loss_recon.item():.4f}'})

    avg_loss = np.mean(test_losses)
    print(f"\n✅ Test Complete. Avg Reconstruction Loss: {avg_loss:.4f}")


def main(args):
    # --- 设备设置 ---
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    args.device = device
    print(f'Current cuda device: {torch.cuda.current_device()}')

    # --- 目录设置 ---
    saveDir = os.path.join(args.logDir, f"{args.sub}_{args.task}_EVR_Hybrid")
    args.savevoice = os.path.join(saveDir, 'savevoice') # 保持与旧脚本一致的文件夹名

    # --- 加载配置文件 ---
    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if not hasattr(args, 'sample_rate_mel'):
        args.sample_rate_mel = 22050

    # --- 初始化模型组件 ---
    print("🚀 Initializing Models...")
    try:
        config_file_g = os.path.join(args.model_config, 'config_g.json')
        with open(config_file_g) as f:
            h_g = AttrDict(json.load(f))
        model_g = HybridGenerator(h_g).to(device)
    except Exception as e:
        print(f"❌ Error loading Generator: {e}")
        return

    try:
        config_file_d = os.path.join(args.model_config, 'config_d.json')
        with open(config_file_d) as f:
            h_d = AttrDict(json.load(f))
        model_d = Discriminator(h_d).to(device)
    except:
        model_d = None

    try:
        config_file_v = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
        with open(config_file_v) as f:
            h = AttrDict(json.load(f))
        vocoder = model_HiFi(h).to(device)
        vocoder_state = torch.load(args.vocoder_pre, map_location=device)
        if 'generator' in vocoder_state:
            vocoder.load_state_dict(vocoder_state['generator'])
        else:
            vocoder.load_state_dict(vocoder_state)
        vocoder.remove_weight_norm()
    except Exception as e:
        print(f"❌ Error loading Vocoder: {e}")
        return

    try:
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        model_STT = bundle.get_model().to(device)
        args.sample_rate_STT = bundle.sample_rate
        decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    except Exception as e:
        print(f"❌ Error loading STT: {e}")
        return

    # --- 加载训练好的权重 ---
    loc_g = os.path.join(saveDir, 'savemodel', 'BEST_checkpoint_g.pt')
    if args.get_checkpoint_path:
        loc_g = args.get_checkpoint_path
    if not load_checkpoint(model_g, loc_g, device):
        print("❌ Cannot proceed without Generator weights.")
        return

    # --- 数据加载器 ---
    print("🚀 Loading Test Dataset...")
    testset = myDataset(mode=1, data=os.path.join(args.dataLoc, args.sub), task=args.task, recon=args.recon)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- 运行测试 ---
    start_time = time.time()
    generat(args, test_loader, (model_g, model_d, vocoder, model_STT, decoder_STT))
    time_taken = time.time() - start_time
    print(f"⏱️ Total Test Time: {time_taken:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid EVRNet Testing')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000')
    parser.add_argument('--model_config', type=str, default='./models')
    parser.add_argument('--dataLoc', type=str, default='./processed_dataset_1')
    parser.add_argument('--config', type=str, default='./config_myPrivate.json')
    parser.add_argument('--logDir', type=str, default='./TrainResult_EVR3')
    parser.add_argument('--get_checkpoint_path', type=str, default='', help='指定要测试的具体 checkpoint 路径')
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Voice_mel')
    parser.add_argument('--word_label', type=list, default=[
        "my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "hero"
    ])
    args = parser.parse_args()

    try:
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        args.word_index, args.word_length = word_index(args.word_label, bundle)
    except:
        print(" Warning: Cannot compute word_index, using placeholder.")
        args.word_index, args.word_length = {}, {}

    main(args)