# PIKAN noisy-data and collocation studies

Use one copy of `PIKAN_signaling.py` for all studies. Change only the noise level and the
collocation-point block described below.

## 1. Add or change noise

Set the desired fractional noise near the top of `sgn.py`:

```python
noise_level = 0.05  # 0, 0.03, 0.05, 0.07, or 0.10
```

The current code uses independent **uniform multiplicative noise**:

```python
noise_matrix = np.random.uniform(-1, 1, size=data0.shape)
noise_matrix[:, 2] = 0.0
data_noisy = data0 * (1 + noise_level * noise_matrix)
```

Thus `noise_level = 0.10` multiplies each measured value by a random factor
between `0.90` and `1.10`. This is relative uniform noise, not additive or
Gaussian noise. The noisy data are saved as `simulated_data_noisy.npz`.

`X3` is the third state (Python column index `2`). It comes from the simulation
and is intentionally **not perturbed by noise**; setting
`noise_matrix[:, 2] = 0.0` keeps it unchanged.

The noise matrix is generated once when the script starts. Therefore, all five
training runs in one job use the same noisy dataset, while a new job generates
a new realization unless a NumPy seed is fixed.

## 2. Fixed collocation points (`F` study)

For `N_colloc` equally spaced points that remain fixed during training, use:

```python
N_colloc = 200
t_colloc = jnp.linspace(0, 180, N_colloc)[:, None]
```

Pass `t_colloc` to `train_model`. Change only `N_colloc` for studies such as
100, 200, 1000, or 1500 points.

## 3. Random/resampled collocation points (`R` study)

To draw `N_colloc` points without replacement from a finer candidate grid, use
this block inside the `for run_id in range(5):` loop:

```python
N_colloc = 200
t_dense_grid = jnp.linspace(0, 180, 1800)[:, None]
key = jax.random.PRNGKey(run_id * 706)
indices = jax.random.choice(
    key, t_dense_grid.shape[0], shape=(N_colloc,), replace=False
)
t_colloc = t_dense_grid[indices]
```

This gives each run a different random subset, which then stays fixed within
that run. With `replace=False`, `N_colloc` cannot exceed the candidate-grid
size (1800 here).

If points must be resampled at **every optimization step**, pass and split a
changing JAX PRNG key inside the training update. Do not use
`np.random.randint` inside a `jax.jit`-compiled loss, because it may be
evaluated only when JAX traces/compiles the function rather than every epoch.

