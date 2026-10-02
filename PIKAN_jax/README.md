# PIKAN noisy-data and collocation studies

This directory contains two PIKAN training scripts:

- `PIKAN_signaling_F.py`: uses a fixed, uniformly spaced set of collocation
  points throughout training.
- `PIKAN_signaling_R.py`: randomly resamples a new set of collocation points
  at every training epoch.

The model, data, loss weights, optimizer, network architecture, number of
epochs, noise treatment, and output calculations are the same. The only
intended experimental difference is how the collocation points are selected.

Both scripts currently use 100 collocation points by default. Therefore, they
represent the `F-100` and `R-100` configurations, respectively.

## Reported manuscript and SI settings

- The main-text noise-robustness experiments reported in Table S8 used 1000
  fixed collocation points for both PINNs and tanh-cPIKANs. To reproduce the
  PIKAN side of that study, run `PIKAN_signaling_F.py` with 1000 points.
- The PIKAN collocation-strategy comparison is reported in Table S11. Its
  `F-N` columns use `PIKAN_signaling_F.py` with `N` points, and its `R-N`
  columns use `PIKAN_signaling_R.py` with `N` points resampled every epoch.

## 1. Fixed collocation points (`PIKAN_signaling_F.py`)

The fixed script uses equally spaced points over `t = [0, 180]`:

```python
t_colloc = jnp.linspace(0, 180, 100)[:, None]
```

The same points are used during every epoch. Change `100` to `200`, `1000`, or
`1500` to obtain the `F-200`, `F-1000`, or `F-1500` configurations.

Fixed points provide deterministic, uniform coverage of time and make repeated
runs easier to compare. However, the physics residual is always evaluated at
the same locations.

## 2. Random collocation points (`PIKAN_signaling_R.py`)

Set the desired number near the top of the random script:

```python
N_colloc = 100
t_dense_grid = jnp.linspace(0, 180, 1800)[:, None]
```

During every optimization update, the script splits a changing JAX random key
and draws `N_colloc` points without replacement from the 1800-point candidate
grid. A new subset is therefore used at every epoch, not only once per run.

Change `N_colloc` to `200`, `1000`, or `1500` for the `R-200`, `R-1000`, or
`R-1500` configurations. Because sampling uses `replace=False`, `N_colloc`
cannot exceed 1800 unless the candidate grid is also enlarged.

Random resampling exposes the physics loss to more time locations over the
course of training and reduces dependence on one fixed grid. It also adds
stochastic variation, so fixed and random methods should be compared using the
same `N_colloc`, noise level, architecture, loss weights, and training epochs.

## 3. Add or change noise

Set the desired fractional noise near the top of either script:

```python
noise_level = 0.05  # 0, 0.03, 0.05, 0.07, or 0.10
```

The code uses independent **uniform multiplicative noise**:

```python
noise_matrix = np.random.uniform(-1, 1, size=data0.shape)
noise_matrix[:, 2] = 0.0
data_noisy = data0 * (1 + noise_level * noise_matrix)
```

Thus, `noise_level = 0.10` multiplies each measured value by a random factor
between `0.90` and `1.10`. This is relative uniform noise, not additive or
Gaussian noise. The noisy data are saved as `simulated_data_noisy.npz`.

`X3` is the third state (Python column index `2`). It comes from the DPD
simulation and is intentionally **not perturbed by noise**. Setting
`noise_matrix[:, 2] = 0.0` keeps it unchanged.

The noise matrix is generated once when the script starts. Therefore, all five
training runs in one job use the same noisy dataset. A new job generates a new
noise realization unless a NumPy seed is fixed.


