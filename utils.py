#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

from __future__ import annotations
import warnings ; warnings.simplefilter('ignore')

import sys
from time import time
import json
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_PATH = Path(__file__).parent.absolute()
DATA_PATH = BASE_PATH / 'data'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
DB_FILE = IMG_PATH / 'run.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'   # for qt model

Model = Module

def seed_everything(seed:int) -> int:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if device == 'cuda':
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def gc_everything():
  for _ in range(2):
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()


Record = Dict[str, Any]
DB = Dict[str, List[Record]]

'''
{
  'model': [{
    ts: int
    cmd: str
    args: {}
    n_layers: int
    acc: float
    racc: float
    pcr: float
    asr: float
  }]
}
'''

def db_load() -> DB:
  if not DB_FILE.exists():
    return {}
  else:
    with open(DB_FILE, 'r', encoding='utf-8') as fh:
      return json.load(fh)

def db_save(db:DB):
  with open(DB_FILE, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False)

def db_add(db:DB, model:str, rec:Record):
  if model in db:
    db[model].append(rec)
  else:
    db[model] = [rec]


def imshow(X:Tensor, AX:Tensor, title:str=''):
  DX = X - AX
  DX = (DX - DX.min()) / (DX.max() - DX.min())

  grid_X  = make_grid( X).permute([1, 2, 0]).detach().cpu().numpy()
  grid_AX = make_grid(AX).permute([1, 2, 0]).detach().cpu().numpy()
  grid_DX = make_grid(DX).permute([1, 2, 0]).detach().cpu().numpy()
  plt.subplot(131) ; plt.title('X')  ; plt.axis('off') ; plt.imshow(grid_X)
  plt.subplot(132) ; plt.title('AX') ; plt.axis('off') ; plt.imshow(grid_AX)
  plt.subplot(133) ; plt.title('DX') ; plt.axis('off') ; plt.imshow(grid_DX)
  plt.tight_layout()
  plt.suptitle(title)

  mng = plt.get_current_fig_manager()
  mng.window.showMaximized()    # 'QT4Agg' backend
  plt.show()


def float_to_str(x:str, n_prec:int=3) -> str:
  # integer
  if int(x) == x: return str(int(x))
  
  # float
  sci = f'{x:e}'
  frac, exp = sci.split('e')
  
  frac_r = round(float(frac), n_prec)
  frac_s = f'{frac_r}'
  if frac_s.endswith('.0'):   # remove tailing '.0'
    frac_s = frac_s[:-2]
  exp_i = int(exp)
  
  if exp_i != 0:
    # '3e-5', '-1.2e+3'
    return f'{frac_s}e{exp_i}'
  else:
    # '3.4', '-1.2'
    return f'{frac_s}'
