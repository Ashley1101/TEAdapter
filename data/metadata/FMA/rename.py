import os
import json

def change_extension(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data["data"]:
        # 获取文件名和后缀
        base, ext = os.path.splitext(entry["wav"])
        # 如果后缀是.mp3，替换为.wav
        if ext.lower() == ".mp3":
            entry["wav"] = base + ".wav"

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 用法示例
change_extension('data/metadata/FMA/datafiles/FMA-intro_melody_train_label.json')
change_extension('data/metadata/FMA/datafiles/FMA-intro_melody_val_label.json')
change_extension('data/metadata/FMA/datafiles/FMA-chorus_melody_train_label.json')
change_extension('data/metadata/FMA/datafiles/FMA-chorus_melody_val_label.json')
change_extension('data/metadata/FMA/datafiles/FMA-outro_melody_train_label.json')
change_extension('data/metadata/FMA/datafiles/FMA-outro_melody_val_label.json')