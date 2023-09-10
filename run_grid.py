#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/10

from run import *
import seaborn as sns

MODELS = [
  'resnet50',
  'wide_resnet50_2',
  'resnext50_32x4d',
  'densenet121',
  'regnet_x_8gf',
  'regnet_y_8gf',
  'convnext_base',
  'mnasnet0_75',
  'mnasnet1_3',
  'mobilenet_v2',
  'mobilenet_v3_large',
  'squeezenet1_1',
  'shufflenet_v2_x2_0',
  'efficientnet_b0',
  'efficientnet_v2_m',
  'swin_b',
  'swin_v2_b',
  'vit_b_16',
  'vit_b_32',
  'maxvit_t',
  'googlenet',
  'inception_v3',
]

R_Ws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@timer
def run_grid(args):
  res = np.zeros([len(MODELS), len(R_Ws)], dtype=np.float32)

  for i, model in enumerate(MODELS):
    for j, r_w in enumerate(R_Ws):
      args.model = model
      args.r_w = r_w

      try:
        rec = run(args, save_rec=False)
        res[i, j] = rec['acc']
      except:
        res[i, j] = -1

  plt.figure(figsize=(12, 16))
  sns.heatmap(res, vmin=0.0, vmax=1.0, annot=True, cbar=True)
  plt.xticks(range(len(R_Ws)), R_Ws)
  plt.yticks(range(len(MODELS)), MODELS)
  plt.savefig(f'run_grid.png')


if __name__ == '__main__':
  run_grid(get_args())
