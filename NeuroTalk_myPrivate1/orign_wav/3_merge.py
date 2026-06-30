from pydub import AudioSegment
from pydub.playback import play

# 加载所有的.wav文件
files = ["my.wav", "dad.wav", "is.wav", "a.wav", "policeman.wav", "he.wav", "will.wav", "always.wav", "become.wav",
         "my.wav", "hero.wav"]

# 创建一个空的AudioSegment对象用于最终输出
output = AudioSegment.silent(duration=10000)  # 初始没有静音

# 目标长度为1.5秒（1500毫秒）
target_length = 1500

# 存储每个音频文件的长度
file_lengths = {}

# 遍历文件列表
for file in files:
    audio = AudioSegment.from_wav(file)
    file_lengths[file] = len(audio)  # 存储当前音频文件的长度

    # 如果音频长度小于目标长度，则补充静音
    if len(audio) < target_length:
        padding = AudioSegment.silent(duration=target_length - len(audio))
        audio = audio + padding

    # 对每个单词重复5次
    for i in range(5):
        repeated_audio = audio
        if i == 4:
            output += repeated_audio + AudioSegment.silent(duration=10000)  # 不同单词间10秒静音
        else:
            # 将重复后的音频添加到输出中
            output += repeated_audio + AudioSegment.silent(duration=5000)  # 单词间5秒静音

# 删除最后一个多余的10秒静音
output = output[:-10000]

# 输出最终的音频文件
output.export("my dad is a policeman he will always become my hero.wav", format="wav")

# 打印每个文件的长度
for file, length in file_lengths.items():
    print(f"{file} length: {length} milliseconds")