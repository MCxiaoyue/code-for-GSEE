import os
from scipy.io import wavfile


def get_duration_scipy(file_path):
    try:
        # scipy 可以读取 float32 格式的 wav
        rate, data = wavfile.read(file_path)
        # 时长 = 帧数 / 采样率
        duration = len(data) / float(rate)
        return duration
    except Exception as e:
        return f"错误: {e}"


# 下面的 format_time 和 main 函数逻辑与方法一相同，只需替换 get_duration 函数即可
def format_time(seconds):
    if isinstance(seconds, str): return seconds
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def main():
    current_dir = os.getcwd()
    wav_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.wav')]

    if not wav_files:
        print("当前文件夹内没有找到 .wav 文件。")
        return

    print(f"共找到 {len(wav_files)} 个 wav 文件：\n{'-' * 60}")
    print(f"{'文件名':<40} | {'时长 (秒)':<10} | {'格式化时长'}")
    print(f"{'-' * 60}")

    total_duration = 0
    valid_count = 0

    for filename in wav_files:
        file_path = os.path.join(current_dir, filename)
        if os.path.isfile(file_path):
            duration = get_duration_scipy(file_path)

            if isinstance(duration, (int, float)):
                total_duration += duration
                valid_count += 1
                print(f"{filename:<40} | {duration:<10.4f} | {format_time(duration)}")
            else:
                print(f"{filename:<40} | 错误       | {duration}")

    print(f"{'-' * 60}")
    if valid_count > 0:
        print(f"有效文件总数: {valid_count}")
        print(f"总时长: {total_duration:.4f} 秒 ({format_time(total_duration)})")


if __name__ == "__main__":
    main()