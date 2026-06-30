import pandas as pd
import os
# 指向你的一个 Mel CSV 文件
file_path = "./processed_dataset/sub1/train/Voice_mel/sub-01_mel_ts-0001.csv"
df = pd.read_csv(file_path, header=None)
print(f"Min: {df.values.min()}, Max: {df.values.max()}, Mean: {df.values.mean()}")