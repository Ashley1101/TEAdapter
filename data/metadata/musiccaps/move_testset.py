import pandas as pd
import os
import shutil

# 读取CSV文件
csv_file_path = 'data/metadata/musiccaps/musiccaps_test.csv'
df = pd.read_csv(csv_file_path)

# 原始音乐文件夹路径
original_music_folder = 'data/musiccaps/audios'

# 新的文件夹路径，用于存储测试集音乐
test_music_folder = 'data/musiccaps/testset_audios'

# 创建新文件夹
os.makedirs(test_music_folder, exist_ok=True)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    # 获取音乐文件名（idx列）
    music_filename = row['ytid']
    
    # 构建原始音乐文件路径
    original_music_path = os.path.join(original_music_folder, music_filename+".mp3")
    
    # 构建目标音乐文件路径
    target_music_path = os.path.join(test_music_folder, music_filename+'.mp3')
    
    # 移动文件
    shutil.copy(original_music_path, target_music_path)


print("Testset created success.")
