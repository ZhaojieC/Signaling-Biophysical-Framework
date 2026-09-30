#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
from scipy import io
import learner as ln
from learner.utils import mse, grad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Module-level copies of the model constants (moved out of run_trial/main so that
# the Monte-Carlo driver and the plotting script share one definition).
K_INDEX = [7, 8, 10, 13, 14, 15, 16, 22]
K_LIST = [1.0, 1.0, 1.0, 1.0, 1.0, 2.3, 2.3, 0.4, 2.0, 2.0,
          0.5, 0.52, 3.8, 0.51, 1.0, 0.7, 2.0, 1.0, 3.0, 0.4,
          0.4, 3.0, 0.6]
K_NORM = [1e6, 1, 1, 1, 1, 1e-4, 1e-4, 1, 1e-2, 1e-6,
          1e-3, 1e-6, 1e-8, 1, 1e-3, 1e-2, 1e-3, 1e-3, 1e-2, 1,
          1, 1, 1e5]
X_INDEX = [0, 2, 3, 4, 7, 8, 9, 10, 13, 14]

class RBCData(ln.Data):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.__init_data()
        
    def generate(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        t = io.loadmat('data/t2.mat')['t_s1_v1']
        x = io.loadmat('data/x2.mat')['out_s1_v1']
        
        noise = np.random.normal(1, 0.05, x.shape)
        x = x * noise
        
        norm = np.max(x, axis=0)[None,:] + 1e-8
        return t, x/norm, norm
        
    def __init_data(self):
        self.X_train, self.y_train, self.norm = self.generate()
        self.X_test, self.y_test, _ = self.generate()

class RBCPINN(ln.nn.LossNN):
    def __init__(self, net, norm, k_index, k_list, k_norm, x_index, lam1 = 1, lam2 = 1):
        super(RBCPINN, self).__init__()
        self.net = net
        self.norm = norm
        self.k_index = k_index
        self.k_list = k_list
        self.k_norm = k_norm
        self.x_index = x_index
        self.lam1 = lam1
        self.lam2 = lam2
        self.k_dim = 23
        self.K = self.__init_params()
        
    def criterion(self, t, x):
        t = t.requires_grad_(True)
        x_pred = self.net(torch.cat([t,torch.exp(-t)],dim=-1))
        x_t = grad(x_pred, t).squeeze()
        MSE2 = mse(x_pred[...,self.x_index], x[...,self.x_index])
        MSE3 = mse(x_pred[[0,-1]], x[[0,-1]])
        x_pred[...,self.x_index] = x[...,self.x_index]
        MSE1 = mse(x_t, self.f(x_pred))
        return self.lam2 * MSE2 + self.lam1 * (MSE1 + MSE3)
    
    def f(self, x):
        dx = []
        K = [torch.exp(k) for k in self.K]
        K = [K[i] * self.k_norm[i] for i in range(self.k_dim)]
        
        norm = torch.tensor(self.norm, dtype = self.dtype, device = self.device)
        x = x * norm
        dx.append(-(K[6]*x[...,0]*x[...,14]-K[20]*x[...,1])-(K[2]*K[5]*x[...,0]*x[...,13]-K[2]*K[19]*x[...,11]))
        dx.append((K[6]*x[...,0]*x[...,14]-K[20]*x[...,1]) - (K[7]*x[...,1]*torch.abs(1- x[...,4]/(x[...,4]+K[22]))**K[21]) + K[8]*x[...,2])
        dx.append((K[7]*x[...,1]*torch.abs(1- x[...,4]/(x[...,4]+K[22]))**K[21]) - K[8]*x[...,2])         
        dx.append(-K[15]*x[...,3]*x[...,2] + K[16]*x[...,4])
        dx.append(K[15]*x[...,3]*x[...,2] - K[16]*x[...,4])
        dx.append(K[12]*x[...,4]*x[...,6] - K[11]*x[...,5]*(x[...,12] + x[...,10] + x[...,9]))
        dx.append(-K[12]*x[...,4]*x[...,6] + K[11]*x[...,5]*(x[...,12] + x[...,10] + x[...,9]))
        dx.append(-K[0]*K[1]*K[9]*x[...,7] + K[10]*x[...,9])
        dx.append(K[14]*x[...,10] - K[3]*K[4]*K[13]*x[...,8])
        dx.append(K[0]*K[1]*K[9]*x[...,7] -K[10]*x[...,9])
        dx.append(-K[14]*x[...,10] + K[3]*K[4]*K[13]*x[...,8])
        dx.append(K[2]*K[5]*x[...,0]*x[...,13]-K[2]*K[19]*x[...,11]-K[17]*x[...,11]+K[18]*x[...,12])
        dx.append(K[17]*x[...,11]-K[18]*x[...,12])
        dx.append(-(K[2]*K[5]*x[...,0]*x[...,13]-K[2]*K[19]*x[...,11]))
        dx.append(-(K[6]*x[...,0]*x[...,14] - K[20]*x[...,1]))
        dx = torch.stack(dx, dim = -1) / norm
        return dx
        
    def predict(self, t, returnnp=False):
        x = self.net(torch.cat([t,torch.exp(-t)],dim=-1))
        if returnnp:
            x = x.detach().cpu().numpy()
        return x
    
    def __init_params(self):
        params = torch.nn.ParameterList()
        for i in range(self.k_dim):
            if i not in self.k_index:
                ki = torch.tensor(self.k_list[i], dtype = self.dtype, device = self.device)
                params.append(torch.nn.Parameter(torch.log(ki), requires_grad = False))
            else:
                ki = torch.randn(1, dtype = self.dtype, device = self.device)[0]
                params.append(torch.nn.Parameter(ki, requires_grad = True))
        return params

def callback(data, net):
    t = data.X_train
    x = data.y_train
    t = t.requires_grad_(True)
    x_pred = net.net(torch.cat([t,torch.exp(-t)],dim=-1))
    x_t = grad(x_pred, t).squeeze()
    MSE2 = mse(x_pred[...,net.x_index], x[...,net.x_index])
    MSE3 = mse(x_pred[[0,-1]], x[[0,-1]])
    x_pred[...,net.x_index] = x[...,net.x_index]
    MSE1 = mse(x_t, net.f(x_pred))
    return [MSE1.item(), MSE2.item(), MSE3.item()]

def run_trial(seed, trial_num, iterations=100000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    data = RBCData(seed=seed)
    
    k_index, k_list, k_norm, x_index = K_INDEX, K_LIST, K_NORM, X_INDEX

    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    fnn = ln.nn.FNN(2, 15, 5, width=60, activation='tanh')
    fnn.apply(weight_init)
    
    net = RBCPINN(fnn, data.norm, k_index, k_list, k_norm, x_index, lam1=0.1, lam2=1)
    
    args = {
        'data': data,
        'net': net,
        'criterion': None,
        'optimizer': 'adam',
        'lr': 0.001,
        'iterations': iterations,
        'print_every': 10000,
        'save': False,
        'callback': callback,
        'path': f'trial_{trial_num}',
        'dtype': 'float',
        'device': 'cpu',
    }
    
    ln.Brain.Init(**args)
    ln.Brain.Run()
    
    with torch.no_grad():
        t_test = torch.tensor(data.X_test, dtype=torch.float32)
        y_pred = net.predict(t_test, returnnp=True) * data.norm
    
    k_learned = []
    for i in k_index:
        k_val = torch.exp(net.K[i]).item() * net.k_norm[i]
        k_learned.append(k_val)
    
    return y_pred, np.array(k_learned)

def plot_uncertainty(data, all_preds):
    t = data.X_test.squeeze()
    y_true = (data.y_test * data.norm).squeeze()
    
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    min_pred = np.min(all_preds, axis=0)
    max_pred = np.max(all_preds, axis=0)
    
    name_list = ['SIRPa','CD47.SIRPa','CD47.SIRPa_act','SHP1_inact','SHP1_act',
                 'Myo2a_inact','Myo2a_act','FCR','PSR','FCR_act','PSR_act',
                 'TSP1.CD47.SIRPa','TSP1.CD47.SIRPa_act','CD47_alt','CD47']

    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 5, hspace=0.4, wspace=0.3)
    
    for i in range(15):
        ax = plt.subplot(gs[i//5, i%5])
        
        # Plot confidence bounds first
        ax.fill_between(t.squeeze(), 
                        (mean_pred[:, i] - 2*std_pred[:, i]), 
                        (mean_pred[:, i] + 2*std_pred[:, i]), 
                        alpha=0.3, edgecolor='gray', facecolor='cyan',
                        label="2σ Confidence")
        
        # Plot min-max range
        ax.fill_between(t.squeeze(),
                        min_pred[:, i],
                        max_pred[:, i],
                        alpha=0.2, facecolor='lightgray',
                        label="Min-Max Range")
        
        # Plot mean prediction
        ax.plot(t, mean_pred[:, i], '-r', lw=1.5, label="Mean Prediction")
        
        # Plot true values
        ax.plot(t, y_true[:, i], '--b', lw=1, label="True Solution")
        
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Concentration', fontsize=8)
        ax.set_title(name_list[i], fontsize=10)
        ax.legend(frameon=False, loc='best', fontsize=6)
        ax.grid(alpha=0.2)
        
    plt.tight_layout()
    plt.savefig('uncertainty_results.png', dpi=300, bbox_inches='tight')

def main():
    num_trials = 10
    base_seed = 42
    seeds = [base_seed + i*100 for i in range(num_trials)]
    
    all_preds = []
    all_k = []
    
    for trial in range(num_trials):
        print(f"\n=== Running Trial {trial+1}/{num_trials} ===")
        preds, k_vals = run_trial(seeds[trial], trial+1)
        all_preds.append(preds)
        all_k.append(k_vals)
    
    all_preds = np.array(all_preds)
    all_k = np.array(all_k)
    
    data = RBCData(seed=seeds[0])
    plot_uncertainty(data, all_preds)
    
    # Plot parameter statistics
    k_index, k_list, k_norm = K_INDEX, K_LIST, K_NORM
    true_k = [k_list[i] * k_norm[i] for i in k_index]
    k_names = [f'K{i}' for i in k_index]
    
    plt.figure(figsize=(10,6))
    x_pos = np.arange(len(k_names))
    means = np.mean(all_k, axis=0)
    stds = np.std(all_k, axis=0)
    
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10, label='Estimated')
    plt.scatter(x_pos, true_k, color='red', zorder=3, label='True Value', s=100)
    plt.xticks(x_pos, k_names)
    plt.ylabel('Parameter Value (log scale)')
    plt.yscale('log')
    plt.title('Parameter Estimates with SD vs True Values')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('parameter_statistics.png', dpi=300)

if __name__ == '__main__':
    main()