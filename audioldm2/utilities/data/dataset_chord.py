import json
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torch
# from basicsr.utils import img2tensor


class ChordDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        super(ChordDataset, self).__init__()

        # 读取包含 idx、ytid、labels、caption 的 CSV 文件
        df = pd.read_csv(csv_file)

        self.files = []
        for idx, row in df.iterrows():
            ytid = row['ytid']
            # labels = row['aspect_list']
            caption = row['caption']

            audio_path = os.path.join(root_dir, f"audio/musiccaps/{ytid}.mp3")
            chord_audio_path = os.path.join(root_dir, f"output/audio/{ytid}.mp3")

            self.files.append({'audio_path': audio_path, 'chord_audio_path': chord_audio_path, 'sentence': caption})

    def __getitem__(self, idx):
        file = self.files[idx]

        audio, _ = torchaudio.load(file['audio_path'])
        
        chord_audio, _ = torchaudio.load(file['chord_audio_path'])

        return {'audio': audio, 'chord_auido': chord_audio, 'sentence': file['sentence']}

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    train_dataset = ChordDataset('audioldm2/latent_diffusion/data/musiccaps_train.csv', 'audioldm2/latent_diffusion/modules/extra_condition/Chord_Progressions/assets')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True,)
    
    for _, data in enumerate(train_dataloader):
        print(data.shape)
