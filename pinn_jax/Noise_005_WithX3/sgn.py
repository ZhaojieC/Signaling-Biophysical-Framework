# --- Imports ---
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
from jax import jacrev
from matplotlib import cm
import matplotlib as mpl
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np

t_dense = jnp.linspace(0, 180, 300)[:, None]

noise_level = 0.05

# Load original data
loaded = np.load("simulated_data.npz")
data0 = loaded["data"]
t_i = loaded["t_i"]
IC = loaded["IC"]
max_scl = loaded["max_scl"]


# Generate noise matrix in [-1, 1]
noise_matrix = np.random.uniform(-1, 1, size=data0.shape)

# Do not apply noise to column 2 (X3), so set noise to 0 there
noise_matrix[:, 2] = 0.0

# Apply multiplicative noise to all except X3
data_noisy = data0 * (1 + noise_level * noise_matrix)

# Save the noisy data
np.savez("simulated_data_noisy.npz", data=data_noisy, t_i=t_i, IC=IC, max_scl=max_scl)

# Load and read X3 data for plotting or evaluation (optional)
df = pd.read_csv("X3_data.csv")
t_x3 = df["t"].values
x3_vals = df["X3_data"].values

# Reload noisy data (if needed downstream)
loaded = np.load("simulated_data_noisy.npz")
data = loaded["data"]
t_i = loaded["t_i"]
IC = loaded["IC"]
max_scl = loaded["max_scl"]


# --- Trainable and fixed K constants ---
k_index = [7, 8, 10, 13, 14, 15, 16, 22]


k_norm = [1e6, 1, 1, 1, 1,
          1e-4, 1e-4, 1, 1e-2, 1e-6,
          1e-3, 1e-6, 1e-8, 1, 1e-3,
          1e-2, 1e-3, 1e-3, 1e-2, 1,
          1, 1, 1e5]

fixed_vals = [
    1.0, 1.0, 1.0, 1.0, 1.0,
    2.3, 2.3, 0.4, 2.0, 2.0,
    0.5, 0.52, 3.8, 0.51, 1.0,
    0.7, 2.0, 1.0, 3.0, 0.4,
    0.4, 3.0, 0.6
]


def extract_K_from_params(params):
    raw_Ks = params[-1].get("Ks", {})
    K_dict = {}

    for i in range(23):
        key = f"K{i+1}"
        if i in k_index:
            raw_val = raw_Ks.get(key, 1.0)
            if key == "K11":
                low, up = 4e-6, 6e-4
                l = (up - low) / 2
                val = l * jnp.tanh(raw_val) + l + low
            elif key == "K23":
                low, up = 5.8e4, 6.2e4
                l = (up - low) / 2
                val = l * jnp.tanh(raw_val) + l + low
            elif key == "K8":
                low, up = 0.3, 0.5
                l = (up - low) / 2
                val = l * jnp.tanh(raw_val) + l + low
            else:
                val = raw_val * k_norm[i]
        else:
            val = fixed_vals[i] * k_norm[i]
        K_dict[key] = val
    return K_dict


x_index = [0, 2, 3, 4, 7, 8, 9, 10, 13, 14]

# --- Model Initialization ---
def init_params(layers, key):
    keys = jax.random.split(key, len(layers) - 1)
    params = []
    for k, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        W = jax.random.normal(k, (n_in, n_out)) / jnp.sqrt(n_in)
        B = jnp.zeros(n_out)
        params.append({'W': W, 'B': B})
    # Add trainable Ks at the end
    Ks = {f"K{i+1}": jnp.array(1.0) for i in k_index}
    params[-1]['Ks'] = Ks
    return params

# --- Forward Model ---
def fwd(params, t):
    X = 0.01 * t
    inputs = X
    *hidden, last = params
    for layer in hidden:
        inputs = jax.nn.swish(inputs @ layer['W'] + layer['B'])
    return inputs @ last['W'] + last['B']

# --- MSE Loss ---
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

# --- Loss Functions ---
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

# --- Training Setup ---
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

# --- Plot Predicted vs Data ---
# --- Plotting ---
def plot_predicted_vs_data(params, t_data, data_noisy, data_true, scale_factor, run_id):
    x_index = [0, 2, 3, 4, 7, 8, 9, 10, 13, 14]  # Only these have data
    pred = fwd(params, t_data)
    num_vars = data_noisy.shape[1]
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(min(num_vars, 15)):
        if i == 2:  # For X3, add actual CSV data points
            axs[i].plot(t_x3, x3_vals, 'o', label="DPD Simulation Data)", markersize=2, color='green' )
            axs[i].legend()

        # True solution (before noise)
        axs[i].plot(t_data, data_true[:, i] * scale_factor[i], 'b-', label='True Solution')

        # Noisy data as scatter — only if this variable is in x_index
        if i in x_index and i != 2:
            axs[i].scatter(t_data, data_noisy[:, i] * scale_factor[i], color='red', s=10, label='Noisy Measured Data', alpha=0.6)

        # Model prediction
        axs[i].plot(t_data, pred[:, i] * scale_factor[i], 'k--', label='PINs Prediction')

        axs[i].set_title(f'X{i+1}')
        axs[i].grid(True)

        if i == 0:
            axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"multi_run_results/predicted_vs_true_run{run_id}.pdf")
    plt.close()

# --- Run Training ---
os.makedirs("multi_run_results", exist_ok=True)
results = []
t_colloc = t_colloc= jnp.linspace(0, 180, 1000)[:, None]
layers = [1, 128, 128, 128, 15]

for run_id in range(8):
    print(f"\n=== Run {run_id+1} ===")
    key = jax.random.PRNGKey(run_id*127)
    params = init_params(layers, key)
    lambdas = [1.0] * 15
    loss_weight = [1.0, 0.1] + [1.0] * 15
    start_time = time.time()
    trained = train_model(params, lambdas, t_i, t_dense, t_colloc, IC, data, max_scl, epochs=100001, loss_weight=loss_weight)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    flat, _ = tree_flatten(trained)
    np.savez(f"multi_run_results/params_run{run_id+1}.npz", *[np.array(p) for p in flat])
    Ks = extract_K_from_params(trained)
    print(Ks)
    result = {"Run": run_id+1}
    for i in range(23):
        k = f"K{i+1}"
        true_val = fixed_vals[i] * k_norm[i]
        inf_val = float(Ks[k])
        rel_error = (inf_val - true_val) / true_val if i in k_index else 0.0
        result[f"Err_{k}"] = rel_error
        result[k] = inf_val
    results.append(result)
    plot_predicted_vs_data(trained, t_dense, data, data0, max_scl, run_id + 1)


# Save CSVs
df = pd.DataFrame(results)
df.to_csv("multi_run_results/summary_K_results.csv", index=False)
# df.mean(numeric_only=True).to_frame().T.to_csv("multi_run_results/average_K_errors.csv", index=False)
error_cols = [col for col in df.columns if col.startswith("Err_")]
df_abs_avg = df[error_cols].abs().mean().to_frame().T
df_abs_avg.to_csv("multi_run_results/average_K_errors.csv", index=False)
print("✅ Done: All results saved to 'multi_run_results/'")
