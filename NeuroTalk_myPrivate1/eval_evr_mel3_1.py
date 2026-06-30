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
import random

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
    test_cer_recon = []
    test_cer_gt = []

    print(f"🚀 Starting Inference on {len(test_loader.dataset)} samples...")

    # --- 确保保存目录存在 ---
    os.makedirs(args.savevoice, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
            input = input.to(args.device)
            target = target.to(args.device)
            target_cl = target_cl.to(args.device)
            # 保留原始语音数据用于对比
            voice_raw = torch.squeeze(voice, dim=-1).to(args.device)
            labels = torch.argmax(target_cl, dim=1)

            # --- 1. 前向传播 & 对齐 ---
            output = model_g(input)
            mel_out = DTW_align(output, target)

            # --- 2. 反归一化 ---
            mean = data_info[0].to(args.device)
            std = data_info[1].to(args.device)

            # 训练代码中，target_denorm 是给 vocoder 用的
            target_denorm = data_denorm(target, mean, std)
            # 训练代码中，mel_out 也是先对齐再反归一化给 vocoder 用的
            output_denorm = data_denorm(mel_out, mean, std)

            # --- 3. Vocoder 转换 (与训练代码对齐) ---
            # 使用与训练代码相同的函数和批次大小
            wav_target = mel2wav_vocoder(torch.unsqueeze(target_denorm[0], dim=0), vocoder, 1)
            wav_target = torch.reshape(wav_target, (len(wav_target), wav_target.shape[-1]))

            wav_recon = mel2wav_vocoder(torch.unsqueeze(output_denorm[0], dim=0), vocoder, 1)
            wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))

            # --- 4. 重采样到 STT 采样率 ---
            wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)
            wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)

            # --- 5. 音频长度对齐 ---
            # 训练代码中有此操作，确保与原始语音长度一致
            if wav_recon.shape[1] != voice_raw.shape[1]:
                p = voice_raw.shape[1] - wav_recon.shape[1]
                p_s = p // 2
                p_e = p - p_s
                wav_recon = F.pad(wav_recon, (p_s, p_e))
            if wav_target.shape[1] != voice_raw.shape[1]:
                p = voice_raw.shape[1] - wav_target.shape[1]
                p_s = p // 2
                p_e = p - p_s
                wav_target = F.pad(wav_target, (p_s, p_e))

            # --- 6. STT 推理 (与训练代码对齐) ---
            # 为生成的音频生成转录文本
            transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, "", 1)  # 训练时此处为真实标签，测试时可为空

            # --- 7. 保存音频文件 (与训练代码 saveData 逻辑对齐) ---
            # 训练代码为每个批次的第一个样本保存一次
            labels_slice = labels[0:1]
            gt_label = args.word_label[labels_slice[0].item()]

            str_tar = gt_label.replace("|", ",").replace(" ", ",")
            str_pred = transcript_recon[0].replace("|", ",").replace(" ", ",")

            title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
            wav_recon_np = np.squeeze(wav_recon.cpu().detach().numpy())
            wavio.write(os.path.join(args.savevoice, f'test_batch_{i}_{title}.wav'), wav_recon_np, args.sample_rate_STT,
                        sampwidth=1)

            # --- 8. 计算并记录损失 ---
            loss_recon = criterion_recon(mel_out, target)
            test_losses.append(loss_recon.item())

            # --- 9. 计算并记录 CER (与训练代码对齐) ---
            # 需要模拟训练代码中的 CER 计算流程
            gt_label_list = []
            for k in range(len(target)):
                gt_label_list.append(args.word_label[labels[k].item()])

            # STT 推理 (与训练代码 saveData 部分对齐)
            # 1. 重采样原始语音
            voice_for_stt = torchaudio.functional.resample(voice_raw, args.sample_rate_mel, args.sample_rate_STT)
            # 2. 长度对齐
            max_len = max(voice_for_stt.shape[1], wav_recon.shape[1])
            voice_for_stt = F.pad(voice_for_stt, (0, max_len - voice_for_stt.shape[1]))
            wav_recon_padded = F.pad(wav_recon, (0, max_len - wav_recon.shape[1]))
            # 3. STT 模型推理
            emission_gt, _ = model_STT(voice_for_stt)
            emission_recon, _ = model_STT(wav_recon_padded)

            # 4. 解码文本
            # decoder_STT_local = GreedyCTCDecoder(labels=model_STT.labels)  # 使用模型的标签
            transcript_gt = []
            transcript_recon_cer = []
            gt_label_text = []

            for j in range(len(labels)):
                trans_gt = decoder_STT(emission_gt[j]).lower()
                trans_recon_single = decoder_STT(emission_recon[j]).lower()

                transcript_gt.append(trans_gt)
                transcript_recon_cer.append(trans_recon_single)
                gt_label_text.append(args.word_label[labels[j].item()].lower())

            # 5. 计算 CER
            cer_recon = CER(transcript_recon_cer, gt_label_text)
            cer_gt = CER(transcript_gt, gt_label_text)  # 理论上接近0

            test_cer_recon.append(cer_recon.item())
            test_cer_gt.append(cer_gt.item())

            pbar.set_postfix({
                'Recon Loss': f'{loss_recon.item():.4f}',
                'CER_recon': f'{cer_recon.item():.4f}'
            })

    avg_loss = np.mean(test_losses)
    avg_cer_recon = np.mean(test_cer_recon) if test_cer_recon else 0.0
    avg_cer_gt = np.mean(test_cer_gt) if test_cer_gt else 0.0

    print(f"\n✅ Test Complete.")
    print(f"   Avg Reconstruction Loss: {avg_loss:.4f}")
    print(f"   Avg CER (Generated Audio): {avg_cer_recon:.4f}")
    print(f"   Avg CER (Ground Truth Audio): {avg_cer_gt:.4f}")


def main(args):
    # --- 设备设置 ---
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    args.device = device
    print(f'Current cuda device: {torch.cuda.current_device()}')

    # --- 目录设置 ---
    saveDir = os.path.join(args.logDir, f"{args.sub}_{args.task}_EVR_Hybrid")
    # 改为 gen_test 以区分于训练时的 epovoice
    args.savevoice = os.path.join(saveDir, 'gen_test')

    # --- 加载配置文件 ---
    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if not hasattr(args, 'sample_rate_mel'):
        args.sample_rate_mel = 22050

    # --- 初始化 STT Bundle (与训练代码对齐) ---
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    args.sample_rate_STT = bundle.sample_rate
    if not hasattr(args, 'word_index'):
        args.word_index, args.word_length = word_index(args.word_label, bundle)

    # --- 初始化模型组件 ---
    print("🚀 Initializing Models...")
    try:
        # Generator
        config_file_g = os.path.join(args.model_config, 'config_g.json')
        with open(config_file_g) as f:
            h_g = AttrDict(json.load(f))
        if not hasattr(h_g, 'upsample_initial_channel'):
            h_g.upsample_initial_channel = 512
        model_g = HybridGenerator(h_g).to(device)
    except Exception as e:
        print(f"❌ Error loading Generator: {e}")
        return

    # Discriminator (加载用于兼容，但测试时不使用)
    try:
        config_file_d = os.path.join(args.model_config, 'config_d.json')
        with open(config_file_d) as f:
            h_d = AttrDict(json.load(f))
        model_d = Discriminator(h_d).to(device)
    except Exception as e:
        print(f"Warning: Could not load Discriminator: {e}")
        model_d = None

    # Vocoder
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
        vocoder.eval()
        for p in vocoder.parameters(): p.requires_grad = False
    except Exception as e:
        print(f"❌ Error loading Vocoder: {e}")
        return

    try:
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        model_STT = bundle.get_model().to(device)
        args.sample_rate_STT = bundle.sample_rate
        # 关键修改：将标签存储在 args 中，以便在 generat 函数中使用
        args.stt_labels = bundle.get_labels()
        decoder_STT = GreedyCTCDecoder(labels=args.stt_labels) # 这里的 decoder_STT 可能暂时用不到
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
    parser.add_argument('--config', type=str, default='./config_myPrivate.json')  # 注意：训练时用的 config_myPrivate1.json
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

    main(args)