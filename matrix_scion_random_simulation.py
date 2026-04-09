import argparse
import itertools
import math
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

parser = argparse.ArgumentParser(description='Muon / Scion steady-state matrix norm simulation')

parser.add_argument('-m', default=384, type=int)
parser.add_argument('-n', default=384, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=0.1, type=float)

HL = 10

eps = 1e-8

# Polar Express (https://arxiv.org/abs/2505.16932) w/ eps to prevent divide-by-zero
coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16() # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim = True) * 1.01 + eps)
    hs = coeffs_list[:steps] + list(itertools.repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1): X = X.mT
    return X

def random_test(m, n, lr, wd, a, nesterov=False, device='cuda'):
  size = (m, n)
  f = f"{m}_by_{n}_matrix_scion_lr_{lr}_wd_{wd}_a_{a}"
  if nesterov:
    f += '_nesterov'
  if os.path.exists(f + '.npy'):
    norms = np.load(f + '.npy')
  else:
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
      o = PolarExpress(m_n, steps=5)
      v.add_(o, alpha=lr)
      norms.append(torch.linalg.matrix_norm(v).item())
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
      norms = random_test(m=args.m, n=args.n, lr=args.lr, wd=args.wd, a=a, nesterov=nesterov, device='mps')
      if nesterov:
        nesterov_mo.append(norms[-1])
      else:
        regular.append(norms[-1])

  plt.scatter(momentum, regular, label="regular")
  plt.scatter(momentum, nesterov_mo, label="Nesterov")

  plt.title(f"${args.m} \\times {args.n}$ matrix norm vs. momentum")
  plt.xlabel("Momentum $\\alpha = 1 - \\mu$")
  plt.ylabel("Norm $||\\theta||_F$")
  plt.legend()

  grid = np.linspace(0.1, 1., num=100)
  pred = np.sqrt(min(args.m, args.n) * args.lr / args.wd / 2 * (2 - grid) / grid)
  nesterov_pred = pred / np.sqrt(1 + 4*grid - 6*grid**2 + 2*grid**3)

  plt.plot(grid, pred)
  plt.plot(grid, nesterov_pred)

  plt.savefig(f"{args.m}_by_{args.n}_matrix_norm_vs_momentum.png")

if __name__ == '__main__':
    main()
