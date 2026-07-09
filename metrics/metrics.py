### We directly copied the metrics.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def signal_class_f1_summary(result_t, result_recall, result_precision):
    """Compute the signal-class F1 confusion matrix from recall and precision.

    The signal-class F1 is the elementwise harmonic mean of the per-task recall
    (``cls_rec``) and precision (``cls_prec``) matrices, i.e.
    ``2 * prec * rec / (prec + rec)`` — it is *not* the tracked ``f1_cls`` metric.
    The same task-boundary reduction as :func:`confusion_matrix` is applied so
    that final F1 and backward transfer (BWT) are directly comparable.

    Args:
        result_t: 1D tensor of task ids, one entry per evaluation round.
        result_recall: 2D tensor (rounds x tasks) of per-task recall scores.
        result_precision: 2D tensor (rounds x tasks) of per-task precision scores.

    Returns:
        A tuple ``(f1_matrix, final_f1, bwt_f1)`` where ``f1_matrix`` is the
        reduced task-by-task F1 matrix, ``final_f1`` is the mean F1 over all
        tasks after training the last task, and ``bwt_f1`` is the mean backward
        transfer of the signal-class F1. Returns ``None`` when the recall and
        precision matrices are empty or their shapes disagree.

    Usage:
        summary = signal_class_f1_summary(result_val_t, result_val_a, result_val_prec)
    """
    if result_recall.numel() == 0 or result_precision.numel() == 0:
        return None
    if result_recall.shape != result_precision.shape:
        return None

    denominator = result_precision + result_recall
    f1_matrix = torch.where(
        denominator > 0,
        2.0 * result_precision * result_recall / denominator,
        torch.zeros_like(denominator),
    )

    number_of_tasks, changes = task_changes(result_t)
    change_indices = torch.LongTensor(changes + [f1_matrix.size(0)]) - 1
    reduced_f1 = f1_matrix[change_indices]

    diagonal_f1 = reduced_f1.diag()
    final_f1 = reduced_f1[number_of_tasks - 1]
    backward_transfer_f1 = final_f1 - diagonal_f1

    return reduced_f1, final_f1.mean(), backward_transfer_f1.mean()


def confusion_matrix(result_t, result_a, log_dir, fname=None):
    nt, changes = task_changes(result_t)
    fname = os.path.join(log_dir, fname)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[(torch.LongTensor(changes))]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, "w")

        print(" ".join(["%.4f" % r for r in baseline]), file=f)
        print("|", file=f)
        for row in range(result.size(0)):
            print(" ".join(["%.4f" % r for r in result[row]]), file=f)
        print("", file=f)
        print("Diagonal Accuracy: %.4f" % acc.mean(), file=f)
        print("Final Accuracy: %.4f" % fin.mean(), file=f)
        print("Backward: %.4f" % bwt.mean(), file=f)
        print("Forward:  %.4f" % fwt.mean(), file=f)
        f.close()

    colors = cm.nipy_spectral(np.linspace(0, 1, len(result)))
    figure = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    # Convert to a plain NumPy array without relying on the deprecated
    # ``copy=`` semantics in NumPy 2.x.
    data = np.asarray(result_a)
    for i in range(len(data[0])):
        plt.plot(
            range(data.shape[0]), data[:, i], label=str(i), color=colors[i], linewidth=2
        )

    plt.savefig(log_dir + "/" + "task_wise_accuracy.png")

    stats = []
    stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats
