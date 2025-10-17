import os
import numpy as np
import matplotlib.pyplot as plt

folder = "logs/ucl_bresnet/ucl_bresnet_test-2025-10-17_13-13-15-8993/0/metrics"
files = sorted(f for f in os.listdir(folder) if f.endswith(".npz"))

# Organize files by task and phase
tasks = sorted(set(f.split('.')[0] for f in files))
phases = ['train', 'prune', 'retrain']

fig, axs = plt.subplots(len(phases), len(tasks), figsize=(6 * len(tasks), 4 * len(phases)), sharex='row')

if len(tasks) == 1:
    axs = np.expand_dims(axs, axis=1)
if len(phases) == 1:
    axs = np.expand_dims(axs, axis=0)

for col, task in enumerate(tasks):
    for row, phase in enumerate(phases):
        fname = f"{task}.npz"
        path = os.path.join(folder, fname)

        if not os.path.exists(path):
            print(f"Missing file: {fname}")
            continue

        data = np.load(path)
        losses = data["losses"]
        accuracies = data["val_acc"]
        print(phase, losses.shape, accuracies.shape)

        ax1 = axs[row, col]
        ax2 = ax1.twinx()

        if losses.ndim == 2:
            ax1.plot(losses[:, 0], label="CE Loss", color='blue')
            ax1.plot(losses[:, 1], label="Mixed Loss", color='purple')
        else:
            ax1.plot(losses, label="Loss", color='blue')

        ax1.set_ylabel("Loss", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.plot(accuracies, label="val_acc", color='green')
        ax2.set_ylabel("val_acc", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        if row == 0:
            ax1.set_title(task)
        if col == 0:
            ax1.set_ylabel("Loss", color='blue')
            ax1.set_xlabel("Epoch")

        ax1.grid(True)

plt.tight_layout()
plt.show()
