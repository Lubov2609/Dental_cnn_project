import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class DentalDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Файл не найден: {img_path}")
        image = Image.open(img_path).convert("RGB")

        labels = self.data.iloc[idx, 1:].astype(float).values
        labels = torch.tensor(labels, dtype=torch.float32)
        print("Loading image:", img_path)
        if self.transform:
            image = self.transform(image)

        return image, labels