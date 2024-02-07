import os

# 读取原始 JSON 文件
import json

def modify_wav_paths(json_file_path, output_file_path):
    # 读取原始 JSON 文件
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # 修改每个数据项的 "wav" 地址
    for item in data['data']:
        old_wav_path = item['wav']
        # 提取文件名
        # filename = old_wav_path.split('/')[-1]
        filename = os.path.splitext(os.path.basename(old_wav_path))[0]
        # 构建新的地址
        new_wav_path = f"melody/audios/{filename}.wav"
        # 更新数据
        item['wav'] = new_wav_path

    # 将修改后的数据写回 JSON 文件
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# 调用函数并传入文件路径
# modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_test_label', 'data/metadata/musiccaps/datafiles/musiccaps_test_label')
# modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_train_label', 'data/metadata/musiccaps/datafiles/musiccaps_train_label')
# modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_val_label', 'data/metadata/musiccaps/datafiles/musiccaps_val_label')

modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_melody_test_label.json', 'data/metadata/musiccaps/datafiles/musiccaps_melody_test_label.json')
modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_melody_train_label.json', 'data/metadata/musiccaps/datafiles/musiccaps_melody_train_label.json')
modify_wav_paths('data/metadata/musiccaps/datafiles/musiccaps_melody_val_label.json', 'data/metadata/musiccaps/datafiles/musiccaps_melody_val_label.json')
