#!/usr/bin/env python3
"""One Monte-Carlo trial for Fig S4. Run as: python mc_trial.py <trial_index>

Saves k_out/k_<idx>.npz holding
    k_abs  : learned parameters in absolute units (k_list * k_norm convention)
    k_rel  : learned parameters divided by k_norm  -> same units as the k_true file
    loss   : final training loss (for filtering non-converged trials)
Seeds follow the original example_rbcUQ3.py convention: seed = 42 + 100*idx,
so trials 0-9 reproduce the original 10-trial run exactly.
"""
import os, sys, time
import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from example_rbcUQ3 import run_trial, RBCData, K_INDEX, K_LIST, K_NORM

idx = int(sys.argv[1])
seed = 42 + idx * 100
out = f"k_out/k_{idx:03d}.npz"
if os.path.exists(out):
    print(f"trial {idx} already done, skipping")
    sys.exit(0)
os.makedirs("k_out", exist_ok=True)

t0 = time.time()
y_pred, k_abs = run_trial(seed, idx, iterations=100000)
k_rel = np.array([k_abs[j] / K_NORM[i] for j, i in enumerate(K_INDEX)])

np.savez(out, k_abs=k_abs, k_rel=k_rel, y_pred=y_pred, seed=seed)
print(f"trial {idx} seed {seed} done in {time.time()-t0:.0f} s")
print("k_rel =", np.array2string(k_rel, precision=4))
