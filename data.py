#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

import os
import json
import random
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from utils import DATA_PATH


class ImageNet_1k(Dataset):

  def __init__(self, root:str, limit:int=-1, shuffle:bool=False):
    self.base_path = os.path.join(root, 'val')

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]
    if shuffle: random.shuffle(self.metadata)
    if limit > 0: self.metadata = self.metadata[:limit]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp).convert('RGB')

    if 'use numpy':
      im = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)   # [C, H, W]
      im = im / np.float32(255.0)
    else:
      im = T.ToTensor()(img)

    return im, tgt


def normalize(X: torch.Tensor) -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)
  return TF.normalize(X, mean, std)       # [B, C, H, W]


def get_dataloader(batch_size=32, limit=-1, shuffle=False):
  root = str(DATA_PATH / 'imagenet-1k')
  dataset = ImageNet_1k(root, limit, shuffle)
  dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0)
  return dataloader
