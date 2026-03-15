import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class DentalDataset(Dataset):

    def __init__(self, csv_file, img_dir, transforms_list):

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms_list = transforms_list

        self.num_aug = len(transforms_list)

    def __len__(self):

        return len(self.data) * self.num_aug

    def __getitem__(self, idx):

        image_idx = idx // self.num_aug
        aug_idx = idx % self.num_aug

        img_name = self.data.iloc[image_idx,0]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        labels = self.data.iloc[image_idx,1:].astype(int).values
        labels = torch.tensor(labels, dtype=torch.long)

        transform = self.transforms_list[aug_idx]

        image = transform(image)

        return image, labels