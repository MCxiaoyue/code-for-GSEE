import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

# 读取 .wav 文件
audio_file = "a.wav"  # 替换为你的音频文件路径
sample_rate, data = wavfile.read(audio_file)

# 如果是双声道音频，只取其中一个声道（例如左声道）
if len(data.shape) > 1:  # 检查是否是立体声
    data = data[:, 0]  # 取第一个声道

# 计算时间轴
duration = len(data) / sample_rate  # 总时长（秒）
time = np.linspace(0., duration, len(data))  # 时间点数组

# 绘制声波图
plt.figure(figsize=(12, 4))
plt.plot(time, data, color='blue', linewidth=1)
# plt.title("Waveform of Audio File")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.grid(True)  # 添加网格线

# 显示图像
plt.show()