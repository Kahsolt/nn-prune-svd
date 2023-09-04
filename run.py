#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/09/04 

from model import MODELS, get_model, get_linear_layers
from data import get_dataloader, normalize
from utils import *


def pgd(model:Model, X:Tensor, Y:Tensor, eps:float=0.03, alpha:float=0.001, steps:int=40) -> Tensor:
  X = X.clone().detach()
  Y = Y.clone().detach()

  AX = X.clone().detach()
  AX = AX + torch.empty_like(AX).uniform_(-eps, eps)
  AX = torch.clamp(AX, min=0.0, max=1.0).detach()

  for _ in tqdm(range(steps)):
    AX.requires_grad = True

    with torch.enable_grad():
      logits = model(normalize(AX))
      loss = F.cross_entropy(logits, Y, reduction='none')
      
    g = grad(loss, AX, grad_outputs=loss)[0]
    AX = AX.detach() + alpha * g.sign()
    delta = torch.clamp(AX - X, min=-eps, max=eps)
    AX = torch.clamp(X + delta, min=0.0, max=1.0).detach()

  # assure valid rgb pixel
  AX = (AX * 255).round().div(255.0)
  return AX


def prune(args, model:Model) -> Model:
  def prune_w(weight:Tensor):
    # TODO: 
    # 1. 用 torch.linalg.svd 分解矩阵 weight = U @ S @ V
    # 2. 舍弃最后 args.r_w 占比个较小的奇异值和对应的向量
    # 3. 返回低秩矩阵 weight' = U' @ S' @ V'
    return weight

  def prune_b(bias:Tensor):
    # TODO: 
    # 1. 将 bias 中绝对值小于 args.r_b 的项直接改成 0.0
    # 1. 将 bias 进行精度舍入到小数点吼 args.n_prec 位
    # 2. 返回稀疏向量 bias'
    return bias

  named_layers = get_linear_layers(model)
  n_layers = len(named_layers)
  if not n_layers: raise ValueError(f'>> model {args.model} has no Linear layers to prune :(')

  print(f'>> n_layers: {n_layers}')
  for name, layer in named_layers:
    print(f'      {name}')
    layer.weight = Parameter(prune_w(layer.weight))
    if hasattr(layer, 'bias'):
      layer.bias = Parameter(prune_b(layer.bias))

  return model


@torch.inference_mode()
def test(model:Model, dataloader:DataLoader) -> float:
  total, correct = 0, 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)

    logits = model(normalize(X))
    pred = logits.argmax(dim=-1)

    total   += len(pred)
    correct += (pred == Y).sum().item()

  return correct / total if total else 0.0


@torch.no_grad()
def test_atk(args, model:Model, dataloader:DataLoader, show:bool=False) -> tuple:
  total, correct = 0, 0
  rcorrect, changed, attacked = 0, 0, 0

  model.eval()
  for X, Y in tqdm(dataloader):
    X, Y = X.to(device), Y.to(device)

    AX = pgd(model, X, Y, args.eps, args.alpha, args.step)

    if show:
      dx = AX - X
      Linf = dx.abs().max(dim=0)[0].mean()
      L1   = dx.abs().mean()
      L2   = dx.square().sum(dim=0).sqrt().mean()
      print(f'Linf: {Linf}')
      print(f'L2: {L2}')

      imshow(X, AX)

    with torch.inference_mode():
      pred    = model(normalize(X)) .argmax(dim=-1)
      pred_AX = model(normalize(AX)).argmax(dim=-1)

    total    += len(pred)
    correct  += (pred    == Y   )             .sum().item()   # clean correct
    rcorrect += (pred_AX == Y   )             .sum().item()   # adversarial still correct
    changed  += (pred_AX != pred)             .sum().item()   # prediction changed under attack
    attacked += ((pred == Y) & (pred_AX != Y)).sum().item()   # clean correct but adversarial wrong

    if show:
      print('Y:', Y, 'pred:', pred, 'pred_AX:', pred_AX)
      print(f'total: {total}, correct: {correct}, rcorrect: {rcorrect}, changed: {changed}, attacked: {attacked}')

  return [
    correct  / total   if total   else 0,   # Clean Accuracy
    rcorrect / total   if total   else 0,   # Remnant Accuracy
    changed  / total   if total   else 0,   # Prediction Change Rate
    attacked / correct if correct else 0,   # Attack Success Rate
  ]


def run(args):
  ''' Model '''
  model = get_model(args.model)

  ''' Prune '''
  if any([args.r_w > 0.0, args.r_b > 0.0, args.n_prec > 0]):
    prune(args, model)

  ''' Data '''
  dataloader = get_dataloader(args.batch_size, args.limit, args.shuffle)

  if args.atk:
    ''' Attack '''
    acc, racc, pcr, asr = test_atk(args, model, dataloader, show=args.show)
    print(f'Clean Accuracy:         {acc:.2%}')
    print(f'Remnant Accuracy:       {racc:.2%}')
    print(f'Prediction Change Rate: {pcr:.2%}')
    print(f'Attack Success Rate:    {asr:.2%}')
  else:
    ''' Test '''
    acc = test(model, dataloader)
    print(f'Clean Accuracy: {acc:.2%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model name')
  parser.add_argument('-B', '--batch_size', type=int, default=32, help='run batch size')
  parser.add_argument('-L', '--limit',      type=int, default=-1, help='limit run sample count')
  parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
  # prune
  parser.add_argument('--r_w',    type=float, default=0.0, help='prune weight svd drop ratio')
  parser.add_argument('--r_b',    type=float, default=0.0, help='prune bias zero-trim threshold')
  parser.add_argument('--n_prec', type=int,   default=-1,  help='prune bias round n_prec')
  # attack
  parser.add_argument('-X', '--atk', action='store_true', help='enable PGD attack')
  parser.add_argument('--eps',   type=float, default=8/255, help='PGD total threshold')
  parser.add_argument('--alpha', type=float, default=1/255, help='PGD step size')
  parser.add_argument('--step',  type=int,   default=10,    help='PGD step count')
  # debug
  parser.add_argument('--show',    action='store_true', help='debug visualize & inspect')
  args = parser.parse_args()

  run(args)
