import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


wandb.login(key='953e958793b218efb850fa194e85843e2c3bd88b')
api = wandb.Api()
head = "linbin/t2i_ablation_arch"
runs = api.runs(head)

all_norm = ['prenorm', 'postnorm', 'mixnorm', 'sandwich']
run_id_dict = {}
for run in runs:
    run_name = [i in run.name for i in all_norm]
    assert sum(run_name) == 1
    run_name = all_norm[run_name.index(True)]
    if run_id_dict.get(run_name, None) is None:
        run_id_dict[run_name] = [run.id]
    else:
        run_id_dict[run_name].append(run.id)


def get_loss_dict(run_name="prenorm"):

    keys = ['train_loss']
    run_ids = run_id_dict[run_name]

    all_history = {}
    for run_id in tqdm(run_ids):
        run = api.run(f"{head}/{run_id}")
        history = run.history(keys=keys) 
        for k, v in history.items():
            v = list(v.values)
            if k not in all_history:
                all_history[k] = v
            else:
                all_history[k].extend(v)
    step = all_history['_step']
    loss = all_history['train_loss']

    return step, loss

def get_grad_norm_dict(run_name="prenorm"):

    keys = [f"grad_norm_block/block_{i}" for i in range(25)]
    run_ids = run_id_dict[run_name]

    all_history = {}
    for run_id in tqdm(run_ids):
        run = api.run(f"{head}/{run_id}")
        history = run.history(keys=keys) 
        for k, v in history.items():
            v = list(v.values)
            if k not in all_history:
                all_history[k] = v
            else:
                all_history[k].extend(v)

    step = all_history['_step']
    grad_norm = list(np.stack(np.array([all_history[k] for k in keys])).reshape(5, 5, -1))

    return step, grad_norm

x_loss_prenorm, y_loss_prenorm = get_loss_dict(run_name="prenorm")
x_loss_postnorm, y_loss_postnorm = get_loss_dict(run_name="postnorm")
x_loss_mixnorm, y_loss_mixnorm = get_loss_dict(run_name="mixnorm")
x_loss_sandwich, y_loss_sandwich = get_loss_dict(run_name="sandwich")

x_grad_prenorm, y_grad_prenorm = get_grad_norm_dict(run_name="prenorm")
x_grad_postnorm, y_grad_postnorm = get_grad_norm_dict(run_name="postnorm")
x_grad_mixnorm, y_grad_mixnorm = get_grad_norm_dict(run_name="mixnorm")
x_grad_sandwich, y_grad_sandwich = get_grad_norm_dict(run_name="sandwich")


# 创建图像
fig, axes = plt.subplots(8, 4, figsize=(28, 28), sharey='row')  # 2x2布局
axes = axes.flatten()  # 将子图数组展平

# 绘制每个子图
titles = ["Prenorm", "Postnorm", "Mixnorm", "Sandwich"]
data_grad_norms = [
    (x_grad_prenorm, y_grad_prenorm),
    (x_grad_postnorm, y_grad_postnorm),
    (x_grad_mixnorm, y_grad_mixnorm),
    (x_grad_sandwich, y_grad_sandwich),
]


for i, title in enumerate(titles):
    x, y = data_grad_norms[i]
    for j in range(5):
        ax = axes[j*4+i]
        print(title, j*4+i, 'ax', i, j)
        ax.scatter(x, y[j].mean(0), alpha=0.7, edgecolors='k')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.0001, 0.1)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Mean {5*j}-{5*(j+1)} layers")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)


j = j + 1
for i, title in enumerate(titles):
    x, y = data_grad_norms[i]
    ax = axes[j*4+i]
    print(title, j*4+i, 'ax', i, j)
    ax.scatter(x, np.array(y).mean(0).mean(0), alpha=0.7, edgecolors='k')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 0.1)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Mean layers")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    
j = j + 1
for i, title in enumerate(titles):
    x, y = data_grad_norms[i]
    y = np.array(y).reshape(25, -1)
    ax = axes[j*4+i]
    print(title, j*4+i, 'ax', i, j)
    ax.scatter(x, y.std(axis=0), alpha=0.7, edgecolors='k')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(0.0001, 0.1)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Std layers")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)


data_loss = [
    (x_loss_prenorm, y_loss_prenorm),
    (x_loss_postnorm, y_loss_postnorm),
    (x_loss_mixnorm, y_loss_mixnorm),
    (x_loss_sandwich, y_loss_sandwich),
]
for j, title in enumerate(titles):
    x, y = data_loss[j]
    ax = axes[-1-3+j]
    print(-1-3+j)
    ax.scatter(x, y, alpha=0.7, edgecolors='k')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.25, 0.35)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# 调整子图布局
plt.tight_layout()

plt.savefig('test.png')
