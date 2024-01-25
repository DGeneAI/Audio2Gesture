import os
import librosa

dir_data = "/root/project/Audio2Gesture/Data/MOCCA/Source/Test_Data/Audio"

# 获取目录下所有音频文件的路径
audio_file_paths = [os.path.join(dir_data, filename) for filename in os.listdir(dir_data) if filename.endswith(".wav")]

# 计算所有音频片段的长度
total_duration = 0.0

for audio_file_path in audio_file_paths:
    # 使用librosa加载音频
    audio, _ = librosa.load(audio_file_path, sr=16000)
    
    # 获取音频的持续时间（秒）
    duration = librosa.get_duration(y=audio)
    
    # 累加到总持续时间
    total_duration += duration

# 输出总持续时间
print("总音频长度（秒）：", total_duration)