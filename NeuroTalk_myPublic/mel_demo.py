import torch
import torchaudio
import torchaudio.transforms as T
import json
import os
from models.models_HiFi import Generator as model_HiFi
from modules import AttrDict

# 配置路径
vocoder_path = './pretrained_model/UNIVERSAL_V1/g_02500000'
config_path = os.path.join(os.path.split(vocoder_path)[0], 'config.json')
audio_path = './orign_wav/policeman.wav' # 替换为你的音频

# 1. 加载配置
with open(config_path) as f:
    h = AttrDict(json.load(f))

# 2. 加载模型
vocoder = model_HiFi(h).cuda()
state_dict = torch.load(vocoder_path, map_location='cpu')
vocoder.load_state_dict(state_dict['generator'])
vocoder.eval()

# 3. 定义 Mel 提取器 (参数必须与 config.json 一致)
mel_transform = T.MelSpectrogram(
    sample_rate=h.sampling_rate,
    n_fft=h.n_fft,
    win_length=h.win_size,
    hop_length=h.hop_size,
    f_min=h.fmin,
    f_max=h.fmax,
    n_mels=h.num_mels,
    power=1,
    normalized=False,
    norm='slaney',
    mel_scale='slaney'
).cuda()

# 4. 加载音频
wav, sr = torchaudio.load(audio_path)
if sr != h.sampling_rate:
    wav = torchaudio.functional.resample(wav, sr, h.sampling_rate)
wav = wav.cuda()

# 5. 提取 Mel (Log-Mel)
with torch.no_grad():
    mel_spec = mel_transform(wav)
    log_mel_spec = torch.log(mel_spec + 1e-9)

print(f"Original Wave Shape: {wav.shape}")
print(f"Extracted Mel Shape: {log_mel_spec.shape}")

# 6. (可选) 验证：用提取的 Mel 还原语音
recon_wav = vocoder(log_mel_spec)
print(f"Reconstructed Wave Shape: {recon_wav.shape}")

# --- 修复开始 ---
# 1. 移到 CPU 并 脱离计算图 (关键修改：添加 .detach())
recon_wav_cpu = recon_wav.detach().cpu()

# 2. 提取第一个样本并去除通道维度
# 形状: [2, 1, 31232] -> [1, 31232] -> [31232]
wav_to_save = recon_wav_cpu[0].squeeze(0)

# 3. 确保是 2D [Channel, Time]
if wav_to_save.dim() == 1:
    wav_to_save = wav_to_save.unsqueeze(0)

print(f"Saving shape: {wav_to_save.shape}")

# 4. 保存
torchaudio.save("policeman_recon.wav", wav_to_save, h.sampling_rate)

print("验证完成：已保存 policeman_recon.wav")