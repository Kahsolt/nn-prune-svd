#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

from torch.nn import Linear
import torchvision.models as M

from utils import *

MODELS = [
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'wide_resnet50_2',
  'wide_resnet101_2',
  
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',

  'googlenet',

  'inception_v3',

  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',
]


def get_model(name:str) -> Model:
  model: Model = getattr(M,  name)(pretrained=True)
  return model.eval().to(device)


def get_param_cnt(model:Model) -> int:
  return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_linear_layers(model:Model) -> List[Tuple[str, Linear]]:
  layers = []
  for name, layer in model.named_modules():
    if isinstance(layer, Linear):
      layers.append((name, layer))
  return layers
