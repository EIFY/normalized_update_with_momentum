import argparse
import math
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

parser = argparse.ArgumentParser(description='Scion steady-state vector norm simulation')

parser.add_argument('-n', default=1024, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.1, type=float)

HL = 10

def random_test(size, lr, wd, a, nesterov=False, device='cuda'):
  f = f"{size}_vector_scion_lr_{lr}_wd_{wd}_a_{a}"
  if nesterov:
    f += '_nesterov'
  if os.path.exists(f + '.npy'):
    norms = np.load(f + '.npy')
  else:
    size = (size,)
    v = torch.zeros(size, device=device)
    norms = [0.]
    half_life = -round(math.log(2) / math.log(1-lr * wd))
    total_steps = -round(HL * math.log(2) / math.log(1-lr * wd))
    m = g = torch.normal(0, 1, size=size, device=device)
    eta = lr * wd
    for _ in range(total_steps):
      m = (1-a) * m + a * g
      v *= (1 - eta)
      if nesterov:
        m_n = (1-a) * m + a * g
      else:
        m_n = m
      v.add_(m_n / torch.linalg.vector_norm(m_n), alpha=lr)
      norms.append(torch.linalg.vector_norm(v).item())
      g = torch.normal(0, 1, size=size, device=device)
    norms = np.array(norms)
    np.save(f, norms)
  print(f"{size=}, {lr=}, {wd=}, {a=}, {nesterov=}, {norms[-1]=}")
  return norms

def main():
  args = parser.parse_args()

  momentum = np.linspace(0.1, 1., num=10)
  regular = []
  nesterov_mo = []

  for nesterov in [False, True]:
    for a in momentum:
      norms = random_test(size=args.n, lr=args.lr, wd=args.wd, a=a, nesterov=nesterov, device='mps')
      if nesterov:
        nesterov_mo.append(norms[-1])
      else:
        regular.append(norms[-1])

  plt.scatter(momentum, regular, label="regular")
  plt.scatter(momentum, nesterov_mo, label="Nesterov")

  plt.title(f"$n = {args.n}$ vector norm vs. momentum")
  plt.xlabel("Momentum")
  plt.ylabel("Norm")
  plt.legend()

  grid = np.linspace(0.1, 1., num=100)
  pred = np.sqrt(args.lr / 2 / args.wd * (2 - grid) / grid)
  nesterov_pred = pred / np.sqrt(1 + 4*grid - 6*grid**2 + 2*grid**3)

  plt.plot(grid, pred)
  plt.plot(grid, nesterov_pred)

  plt.savefig(f"{args.n}_vector_norm_vs_momentum.png")

if __name__ == '__main__':
    main()
