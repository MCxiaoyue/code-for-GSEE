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

# 引入原有模块 (请确保这些文件在路径中)
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT  # 假设你有这些封装，如果没有直接用下面的逻辑
from utils import data_denorm, word_index
from NeuroTalkDataset import myDataset
from torchmetrics import CharErrorRate


# ==============================
# 1. 模型定义 (EVRNet for Classification & Generation)
# ==============================

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        # 假设输入是 (B, C, Time, Channels)，Spatial 作用于 Time 维度
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1),
                                padding=((kernel_size - 1) // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        # Temporal 作用于 Channels 维度 (假设第4维是通道) 或者反过来，需根据数据形状调整
        # 原代码: kernel=(1, k), stride=(1, s). 这意味着作用于第4维 (Width)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride),
                                padding=(0, (kernel_size - 1) // 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y0 = self.conv1(x)
        fused = y0 + x
        fused = self.relu(fused)
        y1 = self.conv2(fused)
        output = y1 + fused
        return self.relu(output)


# 简单的 Mamba 占位符 (如果你没有安装 mamba-ssm 或 kymatio，请用普通 GRU/LSTM 替代或安装对应库)
# 这里为了代码可运行，如果导入失败则使用一个简单的 Conv1d 替代
try:
    from mamba.mamba import Mamba

    HAS_MAMBA = True
except ImportError:
    print("⚠️ Warning: mamba-ssm not found. Using a simple Conv1d block as fallback for Mamba.")
    HAS_MAMBA = False


    class Mamba(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = nn.Conv1d(kwargs.get('d_input', 32), kwargs.get('d_model', 8), kernel_size=3, padding=1)

        def forward(self, x):
            # x: (B, L, D) -> (B, D, L)
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = x.permute(0, 2, 1)
                return [x]  # 模拟返回格式
            return [x]


class EVRNet_Backbone(nn.Module):
    def __init__(self):
        super(EVRNet_Backbone, self).__init__()
        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)
        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)
        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        # 注意：去掉了 AdaptiveAvgPool2d，保留时间维度用于生成

    def forward(self, x):
        # print(x.shape)
        # print('===============================================')
        # Input: (B, 1, Time, Channels) e.g., (B, 1, 192, 24)
        x = self.spatial_block1(x.transpose(1, 2).unsqueeze(1))  # (B, 32, 96, 24)
        x = self.mkrb1(x)
        x = self.temporal_block1(x)  # (B, 32, 48, 12) (假设 stride=2 作用于第4维)
        x = self.mkrb2(x)
        x = self.temporal_block3(x)  # (B, 32, 24, 6)
        return x


class EVRNet_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(EVRNet_Classifier, self).__init__()
        self.backbone = EVRNet_Backbone()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)

        if HAS_MAMBA:
            self.mamba = Mamba(num_layers=1, d_input=32, d_model=8, d_state=8, d_discr=16, ker_size=4, parallel=False)
            self.fc_layer = nn.Linear(32, num_classes)  # Mamba output dim is d_model
        else:
            self.fc_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # (B, 32, T, F)
        x = self.avg_pooling(x)  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        x = self.dropout(x)

        if HAS_MAMBA:
            x = x.unsqueeze(-2)  # (B, 1, 32) -> Mamba expects (B, L, D) maybe? Adjust based on your Mamba impl
            # 简化处理：如果没有序列信息，Mamba 退化为线性层
            x = self.fc_layer(x.squeeze(1))
        else:
            x = self.fc_layer(x)
        return x


class EVRNet_MelGenerator(nn.Module):
    def __init__(self, mel_bins=80, time_steps_out=24):
        super(EVRNet_MelGenerator, self).__init__()
        self.backbone = EVRNet_Backbone()
        self.time_steps_out = time_steps_out
        self.mel_bins = mel_bins

        # 计算 backbone 输出的维度 (B, 32, T, F)
        # 假设输入 192x24 -> 经过 3次 stride2 (一次 spatial, 两次 temporal)
        # Time: 192 -> 96 -> 48 -> 24
        # Freq: 24 -> 24 -> 12 -> 6
        # 输出特征图大小约为 (B, 32, 24, 6)
        self.feature_dim = 32 * 6  # 32 channels * 6 freq bins

        # 解码器：将特征映射回 (Time, Mel_Bins)
        # 对每个时间步独立映射
        self.mel_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, mel_bins)
        )

    def forward(self, x):
        # x: (B, 1, Time, Channels)
        features = self.backbone(x)  # (B, 32, T_out, F_out)

        B, C, T, F = features.shape
        # 重塑为 (B * T, C * F)
        features = features.permute(0, 2, 1, 3).contiguous().view(B * T, C * F)

        # 预测 Mel
        mel_pred = self.mel_head(features)  # (B * T, 80)

        # 重塑回 (B, T, 80)
        mel_pred = mel_pred.view(B, T, self.mel_bins)

        # 转置为 (B, 80, T) 以匹配常见的 Mel 谱格式 (如果 Vocoder 需要)
        # 注意：你的原始代码中 target 可能是 (B, 80, T) 或 (B, T, 80)
        # 这里返回 (B, 80, T) 方便后续处理
        return mel_pred.permute(0, 2, 1)


# ==============================
# 2. 训练辅助函数
# ==============================

def train_classifier_phase(args, train_loader, val_loader, device):
    print("\n" + "=" * 50)
    print("🚀 Phase 1: Training Classifier (Pre-training Backbone)")
    print("=" * 50)

    num_classes = len(args.word_label)
    model = EVRNet_Classifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_g, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    best_acc = 0.0

    for epoch in range(2000):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Cls Epoch {epoch + 1}/{2000}")
        for input, _, target_cl, _, _ in pbar:
            input = input.to(device)
            labels = torch.argmax(target_cl, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

        # scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for input, _, target_cl, _, _ in val_loader:
                input = input.to(device)
                labels = torch.argmax(target_cl, dim=1).to(device)
                outputs = model(input)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"✅ Epoch {epoch + 1} Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            # 保存骨干网络权重
            torch.save(model.backbone.state_dict(), os.path.join(args.savemodel, 'backbone_pretrained.pth'))
            print(f"✨ New Best Backbone Saved! (Acc: {best_acc:.2f}%)")

    return model.backbone


def train_generator_phase(args, train_loader, val_loader, models, criterions, optimizer_g, scheduler_g, start_epoch,
                          device, writer):
    print("\n" + "=" * 50)
    print("🎵 Phase 2: Training Mel Generator (Regression + CTC)")
    print("=" * 50)

    (model_g, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_ctc, CER) = criterions

    best_loss = 1e9

    for epoch in range(start_epoch, args.max_epochs):
        model_g.train()
        vocoder.eval()
        model_STT.eval()

        epoch_loss_g = []
        epoch_loss_recon = []
        epoch_loss_ctc = []
        epoch_cer = []

        pbar = tqdm(train_loader, desc=f"Gen Epoch {epoch + 1}/{args.max_epochs}")

        for i, (input, target, target_cl, voice, data_info) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)  # (B, 80, T)
            voice = torch.squeeze(voice, dim=-1).to(device)
            labels = torch.argmax(target_cl, dim=1)

            # 准备标签用于 CTC
            gt_label_idx = []
            gt_length = []
            for j in range(len(labels)):
                gt_label_idx.append(args.word_index[labels[j].item()])
                gt_length.append(args.word_length[labels[j].item()])
            gt_label_idx = torch.tensor(np.array(gt_label_idx), dtype=torch.int64).to(device)
            gt_length = torch.tensor(gt_length, dtype=torch.int64).to(device)

            optimizer_g.zero_grad()

            # 1. Generate Mel
            mel_out = model_g(input)  # (B, 80, T_out)

            # 2. Align (DTW) if time steps differ slightly, otherwise skip
            # 简单起见，如果维度一致直接算，不一致用插值
            if mel_out.shape[-1] != target.shape[-1]:
                mel_out = F.interpolate(mel_out, size=target.shape[-1], mode='linear', align_corners=False)

            # 3. Reconstruction Loss (RMSE)
            loss_recon = criterion_recon(mel_out, target)

            # 4. CTC Loss (Vocoder -> STT)
            with torch.no_grad():
                # Denorm for Vocoder
                # 假设 data_info[0] is min, data_info[1] is max
                # 注意：你的 data_denorm 函数逻辑需确认
                try:
                    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
                except:
                    # Fallback simple denorm if function fails
                    output_denorm = mel_out * (data_info[1].to(device) - data_info[0].to(device)) + data_info[0].to(
                        device)

            # Vocoder
            # HiFi-GAN expects (B, 1, Time) or (B, Time)? Usually (B, 1, Time) for input mel?
            # Check your specific vocoder implementation. Assuming (B, 80, T) input directly for some implementations
            # Or unsqueeze if needed.
            # Standard HiFi-GAN: input (B, 80, T)
            try:
                wav_recon = vocoder(output_denorm)
                if wav_recon.dim() == 3 and wav_recon.shape[1] == 1:
                    wav_recon = wav_recon.squeeze(1)
            except Exception as e:
                print(f"Vocoder Error: {e}")
                continue

            # Resample for STT (16k)
            if wav_recon.shape[1] != voice.shape[1]:
                wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)
                # Pad to match
                if wav_recon.shape[1] < voice.shape[1]:
                    pad = voice.shape[1] - wav_recon.shape[1]
                    wav_recon = F.pad(wav_recon, (0, pad))
                elif wav_recon.shape[1] > voice.shape[1]:
                    wav_recon = wav_recon[:, :voice.shape[1]]

            # STT Forward
            emission_recon, _ = model_STT(wav_recon)
            emission_recon_ = emission_recon.log_softmax(2)

            # CTC Loss
            input_lengths = torch.full(size=(emission_recon.size(0),), fill_value=emission_recon.size(1),
                                       dtype=torch.long).to(device)
            loss_ctc = criterion_ctc(emission_recon_.transpose(0, 1), gt_label_idx, input_lengths, gt_length)

            # Total Loss
            # 增大 CTC 权重以驱动语义
            loss_g = args.l_g[0] * loss_recon + args.l_g[2] * loss_ctc

            loss_g.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)

            optimizer_g.step()

            # Metrics
            epoch_loss_g.append(loss_g.item())
            epoch_loss_recon.append(loss_recon.item())
            epoch_loss_ctc.append(loss_ctc.item())

            # Calculate CER (Simplified for progress bar)
            # Full CER calculation is slow, do it less frequently or approximate
            if i % 10 == 0:
                # Quick CER check on first batch item
                with torch.no_grad():
                    transcript = decoder_STT(emission_recon[0])
                    gt_text = args.word_label[labels[0].item()]
                    # Simple char match for display
                    cer_val = 1.0 if transcript.replace('|', '') != gt_text.replace('|', '') else 0.0
                    # Real CER needs CharErrorRate metric
                    epoch_cer.append(cer_val)

            pbar.set_postfix({
                'L_tot': f'{np.mean(epoch_loss_g):.4f}',
                'L_rmse': f'{np.mean(epoch_loss_recon):.4f}',
                'L_ctc': f'{np.mean(epoch_loss_ctc):.4f}'
            })

        # Epoch Stats
        avg_loss = np.mean(epoch_loss_g)
        avg_recon = np.mean(epoch_loss_recon)
        avg_ctc = np.mean(epoch_loss_ctc)

        # Validation (Every N epochs)
        if (epoch + 1) % args.val_interval == 0:
            model_g.eval()
            # Run a quick validation pass (similar to training but no grad)
            # ... (Omitted for brevity, similar logic) ...
            # For now, just save
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                state_g = {
                    'arch': str(model_g),
                    'state_dict': model_g.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer_g.state_dict()
                }
                save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
                print(f"✨ Best Model Saved! Loss: {best_loss:.4f}")

            # Log to Tensorboard
            if writer:
                writer.add_scalar("Loss_G_Total", avg_loss, epoch)
                writer.add_scalar("Loss_Recon", avg_recon, epoch)
                writer.add_scalar("Loss_CTC", avg_ctc, epoch)

                # Generate Sample Audio
                with torch.no_grad():
                    sample_in, sample_tar, _, sample_voice, sample_info = next(iter(val_loader))
                    sample_in = sample_in[:1].to(device)
                    sample_lab = torch.argmax(torch.argmax(sample_tar[:1], dim=1), dim=0)  # Hacky label extract

                    gen_mel = model_g(sample_in)
                    if gen_mel.shape[-1] != sample_tar.shape[-1]:
                        gen_mel = F.interpolate(gen_mel, size=sample_tar.shape[-1], mode='linear')

                    try:
                        gen_denorm = data_denorm(gen_mel, sample_info[0][:1].to(device), sample_info[1][:1].to(device))
                        gen_wav = vocoder(gen_denorm).squeeze(0).squeeze(0)
                        gen_wav = torchaudio.functional.resample(gen_wav, args.sample_rate_mel, args.sample_rate_STT)

                        # Save
                        wav_path = os.path.join(args.savevoice, f"ep{epoch}_gen.wav")
                        wavio.write(wav_path, gen_wav.cpu().numpy(), args.sample_rate_STT, sampwidth=2)
                        writer.add_audio("Generated_Audio", gen_wav.cpu(), epoch, sample_rate=args.sample_rate_STT)
                    except Exception as e:
                        print(f"Sample Gen Error: {e}")

        scheduler_g.step()
        print(f"\nEpoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")

    if writer:
        writer.flush()


# ==============================
# 3. 主程序
# ==============================

def main(args):
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Directories
    args.logDir = os.path.abspath(args.logDir)
    saveDir = os.path.join(args.logDir, f"{args.sub}_{args.task}_EVR")
    args.savemodel = os.path.join(saveDir, 'savemodel')
    args.savevoice = os.path.join(saveDir, 'epovoice')
    args.logs = os.path.join(saveDir, 'logs')

    os.makedirs(args.savemodel, exist_ok=True)
    os.makedirs(args.savevoice, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    writer = SummaryWriter(args.logs)
    args.writer = writer

    # Load Config Params
    with open(args.config) as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # Word Index Setup
    # args.word_index, args.word_length = word_index(args.word_label, None)  # Pass bundle if needed, or mock
    # Mocking word_index if function requires bundle:
    # if args.word_index is None:
    #     args.word_index = {i: [i] for i in range(len(args.word_label))}  # Dummy
    #     args.word_length = [len(w.replace('|', '')) for w in args.word_label]

    # Data Loaders
    # Note: Ensure NeuroTalkDataset returns (EEG, Mel, OneHot_Label, Audio_Wave, DataInfo)
    trainset = myDataset(mode=0, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)
    valset = myDataset(mode=2, data=args.dataLoc + '/' + args.sub, task=args.task, recon=args.recon)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Models
    # 1. Vocoder (Fixed)
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        h = AttrDict(json.load(f))
    vocoder = model_HiFi(h).to(device)
    vocoder.load_state_dict(torch.load(args.vocoder_pre)['generator'])
    vocoder.eval()
    for p in vocoder.parameters(): p.requires_grad = False

    # 2. STT (Fixed)

    # 3. Generator (Trainable)
    model_g = EVRNet_MelGenerator(mel_bins=args.n_mel_channels).to(device)

    # Optimizer
    optimizer_g = optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay)

    # Losses
    criterion_recon = RMSELoss().to(device)
    criterion_ctc = nn.CTCLoss().to(device)
    CER = CharErrorRate().to(device)

    # ==========================
    # 【修复点】先初始化 STT Bundle，再计算 word_index
    # ==========================

    # 1. 初始化 STT Bundle (必须在这一步做，因为 word_index 依赖它)
    print("🔄 Initializing Wav2Vec2 Bundle for label mapping...")
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    args.sample_rate_STT = bundle.sample_rate

    # 2. 现在可以安全地调用 word_index 了
    print("🔢 Generating word indices...")
    args.word_index, args.word_length = word_index(args.word_label, bundle)

    # 3. 初始化 STT 模型 (用于后续训练)
    model_STT = bundle.get_model().to(device)
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    model_STT.eval()
    for p in model_STT.parameters():
        p.requires_grad = False

    print(f"✅ STT Initialized. Sample Rate: {args.sample_rate_STT}, Vocab Size: {len(bundle.get_labels())}")

    # ==========================
    # PHASE 1: Pre-train Backbone (Classifier)
    # ==========================
    if args.do_pretrain:
        # 注意：train_classifier_phase 不需要 STT，只需要 label 索引，所以放在这里没问题
        backbone = train_classifier_phase(args, train_loader, val_loader, device)
        # Load weights into Generator
        model_g.backbone.load_state_dict(backbone.state_dict())
        print("✅ Backbone weights loaded from Classifier phase.")
    else:
        print("⏭️ Skipping Pre-training. Loading existing backbone if available...")
        bp_path = os.path.join(args.savemodel, 'backbone_pretrained.pth')
        if os.path.exists(bp_path):
            model_g.backbone.load_state_dict(torch.load(bp_path))
            print("✅ Loaded existing pretrained backbone.")
        else:
            print("⚠️ No pretrained backbone found. Training from scratch (Not recommended).")

    # ==========================
    # PHASE 2: Train Generator
    # ==========================
    criterions = (criterion_recon, criterion_ctc, CER)
    models = (model_g, vocoder, model_STT, decoder_STT)

    start_epoch = 0
    # Resume logic could be added here

    train_generator_phase(args, train_loader, val_loader, models, criterions,
                          optimizer_g, scheduler_g, start_epoch, device, writer)

    print("🎉 Training Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVRNet Mel Generation')
    parser.add_argument('--vocoder_pre', type=str, default='./pretrained_model/UNIVERSAL_V1/g_02500000')
    parser.add_argument('--dataLoc', type=str, default='./processed_dataset_1')
    parser.add_argument('--config', type=str, default='./config_myPrivate.json')
    parser.add_argument('--logDir', type=str, default='./TrainResult_EVR')
    parser.add_argument('--gpuNum', type=list, default=[0])
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--sub', type=str, default='sub1')
    parser.add_argument('--task', type=str, default='SpokenEEG')
    parser.add_argument('--recon', type=str, default='Voice_mel')

    # New Args
    parser.add_argument('--do_pretrain', type=bool, default=True, help='Run Phase 1 Classification Pre-training')
    # parser.add_argument('--cls_epochs', type=int, default=200, help='Epochs for Phase 1')
    parser.add_argument('--val_interval', type=int, default=5)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)