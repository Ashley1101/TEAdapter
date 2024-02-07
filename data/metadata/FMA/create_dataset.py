import os
import json
from sklearn.model_selection import train_test_split

def split_and_save(input_json, train_output, test_output, data_root, test_size=0.2, random_seed=None):
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # 转换为包含相对路径的数据结构
    data_with_paths = [{"wav": os.path.join(data_root, filename), "caption": caption} for filename, caption in data.items()]

    # 使用sklearn的train_test_split函数划分训练集和测试集
    train_data, test_data = train_test_split(data_with_paths, test_size=test_size, random_state=random_seed)

    # 写入训练集
    with open(train_output, 'w', encoding='utf-8') as train_outfile:
        json.dump({"data": train_data}, train_outfile, ensure_ascii=False, indent=2)

    # 写入测试集
    with open(test_output, 'w', encoding='utf-8') as test_outfile:
        json.dump({"data": test_data}, test_outfile, ensure_ascii=False, indent=2)

# 用法示例
# split_and_save('data/metadata/FMA/intro_annotated.json', 'data/metadata/FMA/datafiles/FMA-intro_train_label.json', 'data/metadata/FMA/datafiles/FMA-intro_val_label.json', data_root='intro/', test_size=0.2, random_seed=42)
# split_and_save('data/metadata/FMA/chorus_annotated.json', 'data/metadata/FMA/datafiles/FMA-chorus_train_label.json', 'data/metadata/FMA/datafiles/FMA-chorus_val_label.json', data_root='chorus/', test_size=0.2, random_seed=45)
# split_and_save('data/metadata/FMA/outro_annotated.json', 'data/metadata/FMA/datafiles/FMA-outro_train_label.json', 'data/metadata/FMA/datafiles/FMA-outro_val_label.json', data_root='outro/', test_size=0.2, random_seed=47)
        
split_and_save('data/metadata/FMA/intro_annotated.json', 'data/metadata/FMA/datafiles/FMA-intro_melody_train_label.json', 'data/metadata/FMA/datafiles/FMA-intro_melody_val_label.json', data_root='melody/', test_size=0.2, random_seed=42)
split_and_save('data/metadata/FMA/chorus_annotated.json', 'data/metadata/FMA/datafiles/FMA-chorus_melody_train_label.json', 'data/metadata/FMA/datafiles/FMA-chorus_melody_val_label.json', data_root='melody/', test_size=0.2, random_seed=45)
split_and_save('data/metadata/FMA/outro_annotated.json', 'data/metadata/FMA/datafiles/FMA-outro_melody_train_label.json', 'data/metadata/FMA/datafiles/FMA-outro_melody_val_label.json', data_root='melody/', test_size=0.2, random_seed=47)
