#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

from __future__ import annotations
import warnings ; warnings.simplefilter('ignore')

from argparse import ArgumentParser, Namespace
from pathlib import Path
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
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_PATH = Path(__file__).parent.absolute()
DATA_PATH = BASE_PATH / 'data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpu = 'cpu'   # for qt model

if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

Model = Module


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
