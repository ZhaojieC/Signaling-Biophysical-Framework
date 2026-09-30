
# Physics-Informed Neural Network with Uncertainty Quantification and Parameter Statistics
# Integrated training + plotting pipeline

import time
import os
from jax.tree_util import tree_flatten
import jax
import jax.numpy as jnp
import optax
from jax import jacrev
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import matplotlib as mpl
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.gridspec as gridspec

# === Uncertainty and Parameter Plotting ===
def plot_uncertainty_all_vars(t, all_preds, data_true, scale_factor):
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    name_list = [f'X{i+1}' for i in range(15)]
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 5, hspace=0.4, wspace=0.3)
    for i in range(15):
        ax = plt.subplot(gs[i//5, i%5])
        ax.fill_between(t.squeeze(), (mean_pred[:, i] - 2 * std_pred[:, i]) * scale_factor[i],
                                      (mean_pred[:, i] + 2 * std_pred[:, i]) * scale_factor[i],
                        alpha=0.3, edgecolor='gray', facecolor='cyan', label="±2σ Band")
        ax.plot(t, mean_pred[:, i] * scale_factor[i], '-r', lw=1.5, label="Mean Prediction")
        ax.plot(t, data_true[:, i] * scale_factor[i], '--b', lw=1, label="True Solution")
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.set_title(name_list[i], fontsize=10)
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("multi_run_results/uncertainty_all_variables.png", dpi=300)
    plt.close()

def plot_parameter_statistics(all_k, k_index, k_norm, fixed_vals):
    k_names = [f'K{i+1}' for i in k_index]
    true_k = [fixed_vals[i] * k_norm[i] for i in k_index]
    means = np.mean(all_k, axis=0)
    stds = np.std(all_k, axis=0)
    x_pos = np.arange(len(k_index))
    plt.figure(figsize=(10, 6))
    plt.scatter(x_pos, true_k, color='red', label='True Value', zorder=3, s=60)
    plt.scatter(x_pos, means, color='blue', alpha=0.6, label='Estimated', zorder=2, s=60)
    plt.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='blue', capsize=8, zorder=1)
    plt.xticks(x_pos, k_names, rotation=45)
    plt.yscale('log')
    plt.ylabel('K Value')
    plt.title('Estimated vs True Parameter Values (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("multi_run_results/parameter_statistics.png", dpi=300)
    plt.close()

# === Load data ===
t_dense = jnp.linspace(0, 180, 300)[:, None]
loaded = np.load("simulated_data.npz")
data0 = loaded["data"]
t_i = loaded["t_i"]
IC = loaded["IC"]
max_scl = loaded["max_scl"]

noise_level = 0.05
noise_matrix = np.random.uniform(-1, 1, size=data0.shape)
noise_matrix[:, 2] = 0.0
data_noisy = data0 * (1 + noise_level * noise_matrix)
np.savez("simulated_data_noisy.npz", data=data_noisy, t_i=t_i, IC=IC, max_scl=max_scl)

df = pd.read_csv("X3_data.csv")
t_x3 = df["t"].values
x3_vals = df["X3_data"].values

loaded = np.load("simulated_data_noisy.npz")
data = loaded["data"]
max_scl = loaded["max_scl"]

# === Constants ===
k_index = [7, 8, 10, 13, 14, 15, 16, 22]
k_norm = [1e6, 1, 1, 1, 1, 1e-4, 1e-4, 1, 1e-2, 1e-6,
          1e-3, 1e-6, 1e-8, 1, 1e-3, 1e-2, 1e-3, 1e-3, 1e-2, 1,
          1, 1, 1e5]
fixed_vals = [1.0]*5 + [2.3, 2.3, 0.4, 2.0, 2.0, 0.5, 0.52, 3.8, 0.51, 1.0,
               0.7, 2.0, 1.0, 3.0, 0.4, 0.4, 3.0, 0.6]

def extract_K_from_params(params):
    raw_Ks = params[-1].get("Ks", {})
    K_dict = {}
    for i in range(23):
        key = f"K{i+1}"
        if i in k_index:
            raw_val = raw_Ks.get(key, 1.0)
            #if key == "K11":
            #    low, up = 4e-6, 6e-4
            #    l = (up - low) / 2
            #    val = l * jnp.tanh(raw_val) + l + low
           # elif key == "K23":
           #     low, up = 5.8e4, 6.2e4
           #     l = (up - low) / 2
            #    val = l * jnp.tanh(raw_val) + l + low
            #elif key == "K8":
            #    low, up = 0.3, 0.5
            #    l = (up - low) / 2
            #    val = l * jnp.tanh(raw_val) + l + low
            #else:
            #    val = raw_val * k_norm[i]
            val = raw_val * k_norm[i]
        else:
            val = fixed_vals[i] * k_norm[i]
        K_dict[key] = val
    return K_dict

#x_index = [0, 2, 3, 4, 7, 8, 9, 10, 13, 14]
x_index = [0, 3, 4, 7, 8, 9, 10, 13, 14]

# === Model Definition ===
def init_params(layers, key):
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(n_in)
        B = jnp.zeros(n_out)
        params.append({'W': W, 'B': B})
    Ks = {f"K{i+1}": jnp.array(1.0) for i in k_index}
    params[-1]['Ks'] = Ks
    return params

def fwd(params, t):
    X = 0.01 * t
    inputs = X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.swish(inputs @ layer['W'] + layer['B'])
    return inputs @ last['W'] + last['B']

@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)

# --- ODE Loss Function ---
def ODE_loss(params, t, y_func, max_scl):
    K_vals = extract_K_from_params(params)
    dy_dt = jax.vmap(jacrev(y_func))(t)
    y_vals = jax.vmap(y_func)(t)
    X = jnp.split(y_vals * max_scl, 15, axis=1)
    K = lambda k: K_vals[k]

    dX = [
        - (K('K7')*X[0]*X[14] - K('K21')*X[1]) - (K('K3')*K('K6')*X[0]*X[13] - K('K3')*K('K20')*X[11]),
       (K('K7')*X[0]*X[14] - K('K21')*X[1]) - (K('K8') * X[1] * (1.0 - X[4]/(X[4] + K('K23'))) ** K('K22')) + K('K9')*X[2],
        K('K8') * X[1] * (1.0 - X[4] / (X[4] + K('K23'))) ** K('K22') - K('K9') * X[2],
        -K('K16')*X[3]*X[2] + K('K17')*X[4],
        K('K16')*X[3]*X[2] - K('K17')*X[4],
        K('K13')*X[4]*X[6] - K('K12')*X[5]*(X[12] + X[10] + X[9]),
        -K('K13')*X[4]*X[6] + K('K12')*X[5]*(X[12] + X[10] + X[9]),
        - (K('K1')*K('K2')*K('K10'))*X[7] + K('K11')*X[9],
        K('K15')*X[10] - (K('K4')*K('K5')*K('K14'))*X[8],
        (K('K1')*K('K2')*K('K10'))*X[7] - K('K11')*X[9],
        -K('K15')*X[10] + (K('K4')*K('K5')*K('K14'))*X[8],
        K('K3')*K('K6')*X[0]*X[13] - K('K3')*K('K20')*X[11] - K('K18')*X[11] + K('K19')*X[12],
        K('K18')*X[11] - K('K19')*X[12],
        - (K('K3')*K('K6')*X[0]*X[13] - K('K3')*K('K20')*X[11]),
        - (K('K7')*X[0]*X[14] - K('K21')*X[1])
    ]
    f_model = jnp.concatenate(dX, axis=1)
    dy_dt_scaled = dy_dt
    rhs_scaled = f_model / max_scl
    # residuals = dy_dt_scaled - rhs_scaled.reshape(300, 15, 1)
    residuals = dy_dt_scaled - rhs_scaled.reshape(t.shape[0], 15, 1)

    return jnp.split(residuals, 15, axis=1) 



def loss_fun(params, lambdas, t_i, t_d, t_c, data_IC, data, max_scl):
    y_func = lambda t: fwd(params, t).squeeze()
    residuals = ODE_loss(params, t_c, y_func, max_scl)
    loss_odes = [lambdas[i] * jnp.mean(res**2) for i, res in enumerate(residuals)]
    pred_IC = fwd(params, t_i).reshape(1, -1)
    loss_IC = MSE(data_IC, pred_IC)
    pred_d = fwd(params, t_d)
    loss_data = sum(MSE(data[:, i:i+1], pred_d[:, i:i+1]) for i in x_index)
    return loss_IC, loss_data, *loss_odes

def loss_fun_total(params, lambdas, t_i, t_d, t_c, data_IC, data, loss_weight, max_scl):
    losses = loss_fun(params, lambdas, t_i, t_d, t_c, data_IC, data, max_scl)
    return sum(w * l for w, l in zip(loss_weight, losses))

def train_model(params, lambdas, t_i, t_d, t_c, data_IC, data, max_scl, epochs, loss_weight):
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)
    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fun_total)(
            params, lambdas, t_i, t_d, t_c, data_IC, data, loss_weight, max_scl
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    for epoch in range(epochs):
        params, opt_state, loss = update(params, opt_state)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.2e}")
    return params

# === Run Training ===
os.makedirs("multi_run_results", exist_ok=True)
results = []
all_preds = []
all_k = []

t_colloc = jnp.linspace(0, 180, 1000)[:, None]
layers = [1, 128, 128, 128, 15]

for run_id in range(10):
    print(f"\n=== Run {run_id+1} ===")
    key = jax.random.PRNGKey(42 + run_id * 103)
    params = init_params(layers, key)
    lambdas = [1.0] * 15
    loss_weight = [1.0, 0.1] + [1.0] * 15
    trained = train_model(params, lambdas, t_i, t_dense, t_colloc, IC, data, max_scl, epochs=100001, loss_weight=loss_weight)
    Ks = extract_K_from_params(trained)
    pred = fwd(trained, t_dense)
    flat, _ = tree_flatten(trained)
    np.savez(f"multi_run_results/params_run{run_id+1}.npz", *[np.array(p) for p in flat])
    all_preds.append(pred)
    Ks_vec = [float(Ks[f"K{i+1}"]) for i in k_index]
    all_k.append(Ks_vec)

all_preds = np.array(all_preds)
all_k = np.array(all_k)

plot_uncertainty_all_vars(t_dense, all_preds, data0, max_scl)
plot_parameter_statistics(all_k, k_index, k_norm, fixed_vals)
