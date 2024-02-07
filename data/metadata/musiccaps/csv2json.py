import pandas as pd
import json
import os

os.chdir('data/metadata/musiccaps/')

def csv_to_json(csv_path, output_json_path, audio_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 转换为JSON格式
    json_data = {"data": []}
    for index, row in df.iterrows():
        wav_path = os.path.join(audio_dir, f"{row['ytid']}.mp3")
        labels = row['audioset_positive_labels']
        caption = row['caption']

        entry = {
            "wav": wav_path,
            "labels": labels,
            "caption": caption
        }

        json_data["data"].append(entry)

    # 将JSON数据写入文件
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

# 例子用法：
audio_file = 'melody/audios'
csv_to_json("./musiccaps_train.csv", "./datafiles/musiccaps_melody_train_label.json", audio_file)
csv_to_json("./musiccaps_val.csv", "./datafiles/musiccaps_melody_val_label.json", audio_file)
csv_to_json("./musiccaps_test.csv", "./datafiles/musiccaps_melody_test_label.json", audio_file)

# csv_to_json("./musiccaps_train.csv", "./datafiles/musiccaps_train_label.json", audio_file)
# csv_to_json("./musiccaps_val.csv", "./datafiles/musiccaps_val_label.json", audio_file)
# csv_to_json("./musiccaps_test.csv", "./datafiles/musiccaps_test_label.json", audio_file)
