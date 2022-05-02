from pathlib import Path

import numpy as np
import pytorch_lightning
import torch
import torchvision.models.mobilenetv2
from torch.utils.data import DataLoader

from data_loading import Subset, CelebADataset
from models.mini_models import BetaVAE_H

seed = 123
pytorch_lightning.seed_everything(seed, workers=True)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

train_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])
val_augmentation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
])

alpha = 10.

dataset = CelebADataset(Path('../img_align_celeba'))

perm = np.random.RandomState(seed).permutation(list(range(len(dataset))))
tr_size = 0.8
train_indices = [perm[i] for i in range(int(len(perm) * tr_size))]
val_indices = [perm[i] for i in range(int(len(perm) * tr_size), len(perm))]
dataset.val_indices = set(val_indices)
assert len(set(train_indices) & set(val_indices)) == 0

train = Subset(dataset, train_indices, train_augmentation)
val = Subset(dataset, val_indices, val_augmentation)


print(len(val_indices), len(val_indices) + len(train_indices))

n_epochs = 100
train_batch_size = 64
test_batch_size = 64


def model():
    model_ = BetaVAE_H(c_dim=20)
    return model_


def optimizer(model_):
    d = model_.parameters()
    optim = torch.optim.Adam(d, 1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 85, 95], gamma=0.1)
    return [optim], [sched]


def train_dataloader():
    return DataLoader(train, train_batch_size, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=3)


def val_dataloader():
    return DataLoader(val, test_batch_size, num_workers=4, prefetch_factor=3)


output = Path('results')
output.mkdir(exist_ok=True)

mlflow_target_uri = Path('mlruns')
mlflow_target_uri.mkdir(exist_ok=True)
mlflow_target_uri = str(mlflow_target_uri)
experiment_name = 'VAE'
run_name = 'c_dim=20'

# devices
device = 'cuda:0'
distributed_train = not isinstance(device, str)
world_size = len(device) if distributed_train else None
