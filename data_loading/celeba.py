from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pipe import where
from torch.utils.data import Dataset


class CelebADataset(Dataset):

    def __init__(self, path, transform=None):
        self.paths = list(Path(path).iterdir() | where(lambda x: x.name.endswith('jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        if self.transform is not None:
            img = self.transform(img)
        return {'x': img}


class Subset(Dataset):
    dataset: Dataset

    def __init__(self, dataset: Dataset, indices, transform=None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            x['x'] = self.transform(x['x'])

        return x

    def __len__(self):
        return len(self.indices)
