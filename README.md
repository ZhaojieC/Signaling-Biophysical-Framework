# Signaling–Biophysical Framework

Code for

> Zhaojie Chai, Nazanin Ahmadi Daryakenari, and George E. Karniadakis.
> A Multiscale Signaling–Biophysical Framework Reveals Mechanisms of Macrophage-Mediated RBC Clearance in Sickle Cell and Gaucher Disease.
> *PNAS Nexus* (2026). Preprint: https://doi.org/10.64898/2026.04.20.719505

## `pinn_jax/`: PINN parameter inference (JAX)

Physics-informed neural networks (PINNs) that infer the kinetic parameters of the 15-variable macrophage signaling ODE model from synthetic time-course data. Each folder is one experiment and contains the training script (`QC_Plot5.py`, or `sgn.py` in the earlier experiments) and its training data (`simulated_data.npz`, `simulated_data_noisy.npz`, `X3_data.csv`).

The folder name encodes the experiment:

| Tag | Meaning |
|---|---|
| `Noise_0`, `Noise_005`, `Noise_010` | 0%, 5% or 10% multiplicative noise on the training data |
| `WithX3` / `WithoutX3` | X3 included in / excluded from the observed variables |
| `Range` / `NoRange` | bounded / unbounded search ranges for selected parameters |
| `UQ` | 20 independent training runs |
| `selected` | inference of the parameter subset K8, K9, K11, K14–K17 and K23 |

To run an experiment, `cd` into its folder and run `python QC_Plot5.py` (or `python sgn.py`). The results are written to `multi_run_results/`.

Requirements: JAX, Optax, NumPy, SciPy, pandas and Matplotlib. The results were produced with Python 3.9.

## `learnerUQ/`: Monte Carlo parameter inference for Fig. S4 (PyTorch)

- `learner/`: the PyTorch training package used by the scripts.
- `example_rbcUQ3.py`: PINN training on data with 5% multiplicative noise. Run directly, it performs 10 trials of 100,000 iterations.
- `mc_trial.py`: one Monte Carlo trial. `python mc_trial.py <i>` runs one trial of `example_rbcUQ3.py` (100,000 iterations, seed 42 + 100 × i) and writes `k_out/k_<i>.npz`. Fig. S4 uses i = 0, …, 99.
- `collect_FigS4.py`: collects `k_out/`, excludes diverged runs, draws Fig. S4, and writes `all_k.npy` and the correlation matrix `FigS4_corr.txt`.

Run the scripts from inside `learnerUQ/`. They read the training data from `data/t2.mat` and `data/x2.mat`, which are not included in this repository.

Requirements: PyTorch, NumPy, SciPy and Matplotlib. The Monte Carlo runs used Python 3.10 and PyTorch 2.1.2.

## PIKAN

The PIKAN implementation will be added separately.
