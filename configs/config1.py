from pathlib import Path

import numpy as np
import pytorch_lightning
import torch
import torchvision.models.mobilenetv2
from torch.utils.data import DataLoader

from data_loading import Subset, CelebADataset
from models.mini_models import Discriminator, Generator

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


def get_vae():
    from importlib.util import spec_from_file_location, module_from_spec
    from utils import DictWrapper
    spec = spec_from_file_location(
        'config2',
        str(Path('mlruns/0/73e2fe2aa65344b299e973f1496ec017/artifacts/config2.py').resolve())
    )
    config = module_from_spec(spec)
    spec.loader.exec_module(config)
    config = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
    config = DictWrapper(config)
    from engine import VAEController
    model_ = VAEController.load_from_checkpoint(
        str(Path(
            'mlruns/0/73e2fe2aa65344b299e973f1496ec017/artifacts/checkpoints/0/73e2fe2aa65344b299e973f1496ec017/checkpoints/epoch=98-step=250667.ckpt').resolve()),
        config=config
    )
    return model_.module


def get_discriminator():
    return Discriminator(z_dim=256 + 20, size=64)


def get_generator():
    return Generator(z_dim=256 + 20, size=64)


def optimizer(discriminator, generator):
    l1 = []
    l2 = []
    for model_ in (discriminator, generator):
        d = model_.parameters()
        optim = torch.optim.Adam(d, 1e-4)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 85, 95], gamma=0.1)
        l1.append(optim)
        l2.append(sched)
    return l1, l2


def train_dataloader():
    return DataLoader(train, train_batch_size, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=3)


def val_dataloader():
    return DataLoader(val, test_batch_size, num_workers=4, prefetch_factor=3)


output = Path('results')
output.mkdir(exist_ok=True)

mlflow_target_uri = Path('mlruns')
mlflow_target_uri.mkdir(exist_ok=True)
mlflow_target_uri = str(mlflow_target_uri)
experiment_name = 'GAN'
run_name = '1'

# devices
device = 'cuda:0'
distributed_train = not isinstance(device, str)
world_size = len(device) if distributed_train else None
