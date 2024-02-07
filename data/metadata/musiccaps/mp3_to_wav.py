from pydub import AudioSegment
import os

# 输入文件夹和输出文件夹
input_folder = 'data/musiccaps/testset_audios'
output_folder = 'data/musiccaps/testset_wav'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 循环处理每个MP3文件
for mp3_file in os.listdir(input_folder):
    if mp3_file.endswith('.mp3'):
        mp3_path = os.path.join(input_folder, mp3_file)
        
        # 读取MP3文件
        audio = AudioSegment.from_mp3(mp3_path)
        
        # 设置采样率为16k
        audio = audio.set_frame_rate(16000)
        
        # 构造输出文件路径，将文件扩展名改为.wav
        wav_file = os.path.splitext(mp3_file)[0] + '.wav'
        wav_path = os.path.join(output_folder, wav_file)
        
        # 保存为WAV文件
        audio.export(wav_path, format='wav')
        # print(f"Converted {mp3_file} to {wav_file}.")

print("Conversion completed.")
