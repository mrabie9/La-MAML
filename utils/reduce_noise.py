import numpy as np

task0 = np.load('data/rff/radar/tasks-noisy-radchar/task0_noisy.npz')
x0 = task0['xtr']
y0 = task0['ytr']

task1 = np.load('data/rff/radar/tasks-noisy-radchar/task1_noisy.npz')
x1 = task1['xtr']
y1 = task1['ytr']

def reduce_noise(x, y, noise_level=0.01):
    """
    Simple noise reduction by subsampling y==0.
    """
    noise_mask = y == 0
    noise_y = y[noise_mask]
    signal_y = y[~noise_mask]
    noise_x = x[noise_mask]
    signal_x = x[~noise_mask]

    shuffle_indices = np.random.permutation(len(noise_y))
    noise_y = noise_y[shuffle_indices]
    noise_x = noise_x[shuffle_indices]
    n_reduced = int(len(noise_y) * noise_level)
    noise_y_reduced = noise_y[:n_reduced]
    noise_x_reduced = noise_x[:n_reduced]
    x_denoised = np.concatenate([signal_x, noise_x_reduced], axis=0)
    y_denoised =  np.concatenate([signal_y, noise_y_reduced], axis=0)
    
    return x_denoised, y_denoised

x0_denoised, y0_denoised = reduce_noise(x0, y0, noise_level=0.01)
x1_denoised, y1_denoised = reduce_noise(x1, y1, noise_level=0.01)
print(f"Task 0: Original size {x0.shape[0]}, Denoised size {x0_denoised.shape[0]}")
print(f"Task 0: Original unique counts {np.unique(y0, return_counts=True)}, Denoised unique counts {np.unique(y0_denoised, return_counts=True)}")
print(f"Task 1: Original unique counts {np.unique(y1, return_counts=True)}, Denoised unique counts {np.unique(y1_denoised, return_counts=True)}")
print(f"Task 1: Original size {x1.shape[0]}, Denoised size {x1_denoised.shape[0]}")

np.savez('data/rff/radar/task0.npz', xtr=x0_denoised, ytr=y0_denoised, xte=task0['xte'], yte=task0['yte'])
np.savez('data/rff/radar/task1.npz', xtr=x1_denoised, ytr=y1_denoised, xte=task1['xte'], yte=task1['yte'])