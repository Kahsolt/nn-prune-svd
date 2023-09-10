#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

from torch.nn import Linear
import torchvision.models as M

from utils import *

MODELS = [
  'alexnet',
  
  'vgg11',
  'vgg11_bn',
  'vgg13',
  'vgg13_bn',
  'vgg16',
  'vgg16_bn',
  'vgg19',
  'vgg19_bn',

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

  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',

  'regnet_x_16gf',
  'regnet_x_1_6gf',
  'regnet_x_32gf',
  'regnet_x_3_2gf',
  'regnet_x_400mf',
  'regnet_x_800mf',
  'regnet_x_8gf',
  'regnet_y_128gf',
  'regnet_y_16gf',
  'regnet_y_1_6gf',
  'regnet_y_32gf',
  'regnet_y_3_2gf',
  'regnet_y_400mf',
  'regnet_y_800mf',
  'regnet_y_8gf',

  'convnext_base',
  'convnext_large',
  'convnext_small',
  'convnext_tiny',

  'mnasnet0_5',
  'mnasnet0_75',
  'mnasnet1_0',
  'mnasnet1_3',

  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'squeezenet1_0',
  'squeezenet1_1',

  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',

  'efficientnet_b0',
  'efficientnet_b1',
  'efficientnet_b2',
  'efficientnet_b3',
  'efficientnet_b4',
  'efficientnet_b5',
  'efficientnet_b6',
  'efficientnet_b7',
  'efficientnet_v2_l',
  'efficientnet_v2_m',
  'efficientnet_v2_s',

  'swin_b',
  'swin_s',
  'swin_t',
  'swin_v2_b',
  'swin_v2_s',
  'swin_v2_t',

  'vit_b_16',
  'vit_b_32',
  'vit_h_14',
  'vit_l_16',
  'vit_l_32',
  
  'maxvit_t',

  'googlenet',

  'inception_v3',
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
