'''
Prepare data from training/test:
1. Load data from the xxx dataset, each sample should be formatted as [audio, extra_condition, text_condition]
2. Divide the training set and test set.
'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = 'data/metadata/musiccaps'

def split_and_save_dataset(input_csv, valid_file_path, train_csv, test_csv, val_csv):
    
    with open(valid_file_path, 'r') as file:
        valid_ytid_list = [line.strip() for line in file]

    # 读取原始CSV文件
    df = pd.read_csv(input_csv)

    # 根据validURLs筛选数据
    df = df[df['ytid'].isin(valid_ytid_list)]

    # 根据 balanced_subset 列划分测试集
    test_df = df[df['is_balanced_subset'] == 1]

    # 根据 balanced_subset 列的值进行划分
    train_val_df = df[df['is_balanced_subset'] != 1]
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    # 选择需要的列
    train_df = train_df[['ytid', 'audioset_positive_labels', 'caption']]
    val_df = val_df[['ytid', 'audioset_positive_labels', 'caption']]
    test_df = test_df[['ytid', 'audioset_positive_labels', 'caption']]

    # 为每个数据集添加 idx 列
    train_df.insert(0, 'idx', range(len(train_df)))
    val_df.insert(0, 'idx', range(len(val_df)))
    test_df.insert(0, 'idx', range(len(test_df)))

    # 保存为新的CSV文件
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

# 使用示例

if __name__ == '__main__':
    original_csc_path = os.path.join(DATASET_DIR, 'musiccaps-public.csv')
    train_dataset_path = os.path.join(DATASET_DIR, 'musiccaps_train.csv')
    test_dataset_path = os.path.join(DATASET_DIR, 'musiccaps_test.csv')
    val_dataset_path = os.path.join(DATASET_DIR, 'musiccaps_val.csv')
    valid_file = os.path.join(DATASET_DIR, 'validURLs')
    split_and_save_dataset(original_csc_path, valid_file, train_dataset_path, test_dataset_path, val_dataset_path)


