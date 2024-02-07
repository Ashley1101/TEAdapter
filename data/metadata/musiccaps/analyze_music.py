import os
from pydub.utils import mediainfo
import numpy as np

def analyze_audio_durations(directory_path, output_path):
    audio_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp3', '.wav', '.flac'))]

    durations = []
    with open(output_path, 'w') as output_file:
        for audio_file in audio_files:
            file_path = os.path.join(directory_path, audio_file)
            
            # 获取音频时长（单位：秒）
            audio_info = mediainfo(file_path)
            duration_sec = float(audio_info['duration'])

            durations.append(duration_sec)

            # 如果音频长度大于10秒，将文件名写入输出文件
            if duration_sec > 10:
                output_file.write(audio_file + '\n')

    return np.array(durations)

# 替换为你的音频目录路径和输出文件路径
audio_directory_path = 'data/musiccaps/chord/audios'
output_file_path = "audioldm2/latent_diffusion/modules/extra_condition/Chord_Progressions/assets/output/long_audio_files.txt"

audio_durations = analyze_audio_durations(audio_directory_path, output_file_path)

# 统计百分比和最值
percentiles = [25, 50, 75]  # 你可以根据需要修改百分比
percentile_values = np.percentile(audio_durations, percentiles)

min_duration = np.min(audio_durations)
max_duration = np.max(audio_durations)

print(f"The min duration: {min_duration} s")
print(f"The max duration: {max_duration} s")
for p, value in zip(percentiles, percentile_values):
    print(f"{p} per centage: {value} s")
