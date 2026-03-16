# TODO: hyperparameter tuner

import importlib
import datetime
import argparse
import time
import os
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable

import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from main_multi_task import life_experience_iid
from utils.training_metrics import macro_f1_including_noise, macro_precision_signal_only, macro_recall

def log_state(enabled, message):
    """Print a timestamped state message when state logging is enabled."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[STATE {}] {}".format(timestamp, message))

def _split_labels(y):
    y_det = None
    if isinstance(y, (tuple, list)) and len(y) == 2:
        return y[0], y[1]
    if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 2:
        y_cls = y[:, 0]
        y_det = y[:, 1]
    elif torch.is_tensor(y) and y.dim() == 2 and y.size(1) == 2:
        y_cls = y[:, 0]
        y_det = y[:, 1]
    else:
        y_cls = y
    return y_cls, y_det

def _split_eval_output(output):
    """Return (cls_rec, cls_prec, cls_f1, det, fa). Missing values are None."""
    if isinstance(output, (tuple, list)):
        if len(output) == 5:
            return output[0], output[1], output[2], output[3], output[4]
        if len(output) == 3:
            return output[0], None, None, output[1], output[2]
        if len(output) == 2:
            return output[0], None, None, output[1], None
    return output, None, None, None, None

def _get_det_logits(model, xb, t):
    if hasattr(model, "forward_heads"):
        det_logits, _ = model.forward_heads(xb)
        return det_logits
    if hasattr(model, "net") and hasattr(model.net, "forward_heads"):
        det_logits, _ = model.net.forward_heads(xb)
        return det_logits
    if hasattr(model, "net") and hasattr(model.net, "forward_features") and hasattr(model.net, "forward_detection"):
        feats = model.net.forward_features(xb)
        return model.net.forward_detection(feats)
    return None

def _false_alarm_rate(preds: torch.Tensor, targets: torch.Tensor) -> float:
    neg_mask = targets == 0
    if not neg_mask.any():
        print("Warning: No negative samples in _false_alarm_rate calculation, returning 0.0")
        return 0.0
    neg_targets = targets[neg_mask]             # true noise label
    neg_preds = preds[neg_mask]                 # predicted noise label
    fp = (neg_preds == 1).sum().item()          # predicted noise but actually signal
    tn = (neg_targets == 0).sum().item() - fp   # predicted noise and actually noise
    denom = fp + tn
    return float(fp / denom) if denom > 0 else -1

def _noise_label_for_task(args, task_idx: int, class_counts: List[int] | None = None) -> int | None:
    if class_counts is None:
        class_counts = getattr(args, "classes_per_task", None)
    if class_counts is None:
        return None
    _, offset2 = misc_utils.compute_offsets(task_idx, class_counts)
    return offset2 - 1 # Assume noise label is highest in task


def _labels_to_numpy(labels: object) -> np.ndarray:
    """Return labels as a NumPy array regardless of source container type."""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def _extract_task_labels(task: object) -> np.ndarray | None:
    """Extract raw labels for a task from tuple tasks or loader-backed datasets."""
    if isinstance(task, (list, tuple)) and len(task) == 3:
        return _labels_to_numpy(task[2])

    dataset = getattr(task, "dataset", None)
    if dataset is None:
        return None

    for attribute_name in ("targets", "labels", "y", "ys"):
        if hasattr(dataset, attribute_name):
            return _labels_to_numpy(getattr(dataset, attribute_name))

    dataset_tensors = getattr(dataset, "tensors", None)
    if isinstance(dataset_tensors, (list, tuple)) and len(dataset_tensors) >= 2:
        return _labels_to_numpy(dataset_tensors[1])

    return None


def _infer_class_counts_from_tasks(tasks: List[object]) -> List[int] | None:
    """Infer per-task class counts directly from task labels."""
    inferred_counts: List[int] = []
    for task in tasks:
        task_labels = _extract_task_labels(task)
        if task_labels is None:
            return None
        y_cls, _ = _split_labels(task_labels)
        y_cls_array = np.asarray(y_cls).reshape(-1)
        inferred_counts.append(int(np.unique(y_cls_array).size))
    return inferred_counts


def eval_class_tasks(model, tasks, args):

    model.eval()
    result = []
    class_counts = _infer_class_counts_from_tasks(tasks)
    for t, task_loader in enumerate(tasks):
        correct = 0.0
        total = 0.0
        noise_label = _noise_label_for_task(args, t, class_counts)

        for (i, (x, y)) in enumerate(task_loader):
            y_cls, y_det = _split_labels(y)
            if not torch.is_tensor(y_cls):
                y_cls = torch.as_tensor(y_cls)
            if y_det is not None and not torch.is_tensor(y_det):
                y_det = torch.as_tensor(y_det)
            if args.cuda:
                x = x.cuda()
            _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
            if y_det is not None:
                mask = (y_det == 1)
                if mask.any():
                    correct += (p[mask] == y_cls[mask]).float().sum().item()
                    total += float(mask.sum().item())
            elif noise_label is not None:
                mask = y_cls != noise_label
                if mask.any():
                    correct += (p[mask] == y_cls[mask]).float().sum().item()
                    total += float(mask.sum().item())
            else:
                correct += (p == y_cls).float().sum().item()
                total += float(y_cls.size(0))

        result.append(correct / total if total > 0 else 0.0)
    return result

def eval_tasks(model, tasks, args, specific_task=None, eval_epistemic = False):
    model.eval()
    device = torch.device('cuda' if getattr(args, 'cuda', False) and torch.cuda.is_available() else 'cpu')
    results = []
    prec_results = []
    f1_results = []
    class_counts = _infer_class_counts_from_tasks(tasks)
    if class_counts is None:
        class_counts = getattr(args, "classes_per_task", None)

    if specific_task is not None:
        tasks = [tasks[specific_task]]

    det_results = []
    det_fa_results = []
    det_metrics_active = False
    for i, task in enumerate(tasks):
        t = i
        recalls = []
        precisions = []
        f1s = []
        det_recalls = []
        det_false_alarms = []
        noise_label = _noise_label_for_task(args, t, class_counts)
        for batch in task:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device)
            if getattr(args, "arch", "").lower() == "linear":
                xb = xb.view(xb.size(0), -1)
            yb_cls, yb_det = _split_labels(yb)
            if not torch.is_tensor(yb_cls):
                yb_cls = torch.as_tensor(yb_cls)
            if not getattr(args, "use_detector_arch", False):
                yb_det = None
            elif yb_det is not None and not torch.is_tensor(yb_det):
                yb_det = torch.as_tensor(yb_det)

            logits = model(xb, t) if args.model != "anml" else model(xb, fast_weights=None)
            pb = torch.argmax(logits, dim=1).cpu()
            yb_cls_cpu = yb_cls.detach().cpu()
            yb_det_cpu = yb_det.detach().cpu() if yb_det is not None else None
            yb_cls_for_metrics = yb_cls_cpu
            noise_label_for_metrics = noise_label
            if 'ucl' in args.model:
                offset1, _ = misc_utils.compute_offsets(
                    t, class_counts if class_counts is not None else args.nc_per_task
                )
                if yb_det_cpu is not None:
                    yb_cls_for_metrics = yb_cls_cpu.clone()
                    cls_target_mask = yb_det_cpu == 1
                    if cls_target_mask.any():
                        yb_cls_for_metrics[cls_target_mask] = yb_cls_for_metrics[cls_target_mask] - offset1
                else:
                    yb_cls_for_metrics = yb_cls_cpu - offset1
                    if noise_label_for_metrics is not None:
                        noise_label_for_metrics = noise_label_for_metrics - offset1
            # Record total F1 score for all classes including noise
            if yb_det is None:
                f1s.append(macro_f1_including_noise(pb, yb_cls_for_metrics))
            else:
                print("[WARNING] F1 not supported for detection architecture.")
                f1s.append(0.0)

            if yb_det_cpu is not None:
                cls_mask = yb_det_cpu == 1
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask], yb_cls_for_metrics[cls_mask]))
                    precisions.append(
                        macro_precision_signal_only(
                            pb[cls_mask], yb_cls_for_metrics[cls_mask], noise_label_for_metrics
                        )
                    )
            elif noise_label_for_metrics is not None:
                cls_mask = yb_cls_for_metrics != noise_label_for_metrics
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask], yb_cls_for_metrics[cls_mask]))
                    precisions.append(
                        macro_precision_signal_only(
                            pb[cls_mask], yb_cls_for_metrics[cls_mask], noise_label_for_metrics
                        )
                    )
            else:
                recalls.append(macro_recall(pb, yb_cls_for_metrics))
                precisions.append(
                    macro_precision_signal_only(pb, yb_cls_for_metrics, noise_label_for_metrics)
                )

            if yb_det_cpu is not None:
                det_logits = _get_det_logits(model, xb, t)
                if det_logits is not None:
                    det_pred = (det_logits >= 0).long().cpu()
                    det_recalls.append(macro_recall(det_pred, yb_det_cpu))
                    det_false_alarms.append(_false_alarm_rate(det_pred, yb_det_cpu))
            elif noise_label_for_metrics is not None:
                det_targets = (yb_cls_for_metrics != noise_label_for_metrics).long()
                det_pred = (pb != noise_label_for_metrics).long()
                det_recalls.append(macro_recall(det_pred, det_targets))
                det_false_alarms.append(_false_alarm_rate(det_pred, det_targets))

        results.append(sum(recalls) / len(recalls) if recalls else 0.0)
        prec_results.append(sum(precisions) / len(precisions) if precisions else 0.0)
        f1_results.append(sum(f1s) / len(f1s) if f1s else 0.0)
        if det_recalls:
            det_results.append(sum(det_recalls) / len(det_recalls))
            det_fa_results.append(sum(det_false_alarms) / len(det_false_alarms))
            det_metrics_active = True
        else:
            det_results.append(0.0)
            det_fa_results.append(0.0)

    if det_metrics_active:
        return results, prec_results, f1_results, det_results, det_fa_results
    return results, prec_results, f1_results, None, None

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []
    result_val_prec = []
    result_val_f1 = []
    result_val_det_a = []
    result_test_det_a = []
    result_val_det_fa = []
    result_test_det_fa = []

    result_val_t = []
    result_test_t = []

    last_tr_cls_rec = last_tr_cls_prec = last_tr_cls_f1 = None
    last_tr_det = last_tr_fa = None

    time_start = time.time()
    train_task_loaders = []
    test_task_loaders = []
    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    interactive_terminal = sys.stdout.isatty()
    log_state(args.state_logging, "Life experience start: {} tasks queued".format(inc_loader.n_tasks))

    for task_i in range(inc_loader.n_tasks):
        result_epoch_loss = []
        result_acc_val = []
        result_acc_tr = []
        task_info, train_loader, _, test_loader = inc_loader.new_task()
        train_task_loaders.append(train_loader)
        test_task_loaders.append(test_loader)
        current_task = task_info["task"]

        # Per-epoch training metrics for this task (classification + detection).
        per_epoch_train_cls_rec = []
        per_epoch_train_cls_prec = []
        per_epoch_train_det_rec = []
        per_epoch_train_det_pfa = []
        per_epoch_train_f1 = []

        # Per-evaluation validation metrics for this task (classification + detection).
        per_epoch_val_cls_rec = []
        per_epoch_val_cls_prec = []
        per_epoch_val_det_rec = []
        per_epoch_val_det_pfa = []
        per_epoch_val_f1 = []

        log_state(
            args.state_logging,
            "Starting task {} ({}/{})".format(current_task, task_i + 1, inc_loader.n_tasks),
        )
        for ep in range(args.n_epochs):
            model.real_epoch = ep
            epoch_losses = []
            epoch_train_accs = []
            epoch_precisions = []
            epoch_f1s = []
            epoch_det_recalls = []
            epoch_det_fas = []
            epoch_eval_mode_recalls = []
            epoch_start_time = time.time()
            epoch_eval_time = 0.0
            log_state(
                args.state_logging,
                "Task {} Epoch {}/{}: entering train loop".format(current_task, ep + 1, args.n_epochs),
            )

            prog_bar = tqdm(train_loader, disable=not interactive_terminal)
            for (i, (x, y)) in enumerate(prog_bar):

                v_x = x
                y_cls, y_det = _split_labels(y)
                if not torch.is_tensor(y_cls):
                    y_cls = torch.as_tensor(y_cls)
                if y_det is not None and not torch.is_tensor(y_det):
                    y_det = torch.as_tensor(y_det)
                v_y = (y_cls, y_det) if y_det is not None else y_cls
                if args.cuda:
                    v_x = v_x.cuda()
                    if isinstance(v_y, (tuple, list)) and len(v_y) == 2:
                        v_y = (v_y[0].cuda(), v_y[1].cuda())
                    else:
                        v_y = v_y.cuda()
                model.train()

                loss, cls_tr_rec = model.observe(Variable(v_x), v_y, task_info["task"])
                # debug_noise_label = _noise_label_for_task(args, task_info["task"])
                # model.eval()
                # with torch.no_grad():
                #     debug_logits = (
                #         model.forward_training(v_x, task_info["task"])
                #         if args.model != "anml"
                #         else model(v_x, fast_weights=None)
                #     )
                #     debug_preds = torch.argmax(debug_logits, dim=1).cpu()
                #     debug_y_cls, debug_y_det = _split_labels(v_y)
                #     if torch.is_tensor(debug_y_cls):
                #         debug_y_cls_cpu = debug_y_cls.detach().cpu()
                #     else:
                #         debug_y_cls_cpu = torch.as_tensor(debug_y_cls)
                #     debug_eval_recall = 0.0
                #     if debug_y_det is not None:
                #         debug_y_det_cpu = (
                #             debug_y_det.detach().cpu()
                #             if torch.is_tensor(debug_y_det)
                #             else torch.as_tensor(debug_y_det)
                #         )
                #         debug_mask = debug_y_det_cpu == 1
                #         if debug_mask.any():
                #             debug_eval_recall = macro_recall(
                #                 debug_preds[debug_mask],
                #                 debug_y_cls_cpu[debug_mask],
                #             )
                #     elif debug_noise_label is not None:
                #         debug_mask = debug_y_cls_cpu != debug_noise_label
                #         if debug_mask.any():
                #             debug_eval_recall = macro_recall(
                #                 debug_preds[debug_mask],
                #                 debug_y_cls_cpu[debug_mask],
                #             )
                #     else:
                #         debug_eval_recall = macro_recall(debug_preds, debug_y_cls_cpu)
                #     epoch_eval_mode_recalls.append(float(debug_eval_recall))
                #     if i == 0:
                #         print(
                #             "DEBUG batch0: cls_tr_rec {:.5f} | eval_mode_recall {:.5f} | noise_label {}".format(
                #                 float(cls_tr_rec), float(debug_eval_recall), debug_noise_label
                #             )
                #         )
                # model.train()
                # logits = model(x, task_i) if args.model != 'anml' else model(x, task_i, fast_weights=None)
                # pb = torch.argmax(logits, dim=1)
                # correct += (pb == y).sum().item()
                # cls_tr_rec = correct / x.size(0)
                result_acc_tr.append(cls_tr_rec)
                result_epoch_loss.append(loss)
                epoch_losses.append(loss)
                epoch_train_accs.append(cls_tr_rec)

                # Batch-level precision (signal only) and F1 (all classes incl. noise) for progress bar
                noise_label = _noise_label_for_task(args, task_info["task"])
                model.eval()
                with torch.no_grad():
                    logits = (
                        model(v_x, task_info["task"])
                        if args.model != "anml"
                        else model(v_x, fast_weights=None)
                    )
                    pb = torch.argmax(logits, dim=1).cpu()
                    det_logits = None # _get_det_logits(model, v_x, task_info["task"])
                model.train()

                y_cls_for_metric = y_cls.cpu() if torch.is_tensor(y_cls) else torch.as_tensor(y_cls)
                y_det_for_metric = y_det.cpu() if y_det is not None and torch.is_tensor(y_det) else (torch.as_tensor(y_det) if y_det is not None else None)
                noise_label_for_metric = noise_label

                # For split (task-incremental) models, forward returns task-local logits so pb is in [0, C_t-1].
                # Convert labels to task-local so Train Acc / Prec / F1 match.
                if getattr(model, "split", False):
                    offset1, _ = model.compute_offsets(task_info["task"])
                    y_cls_for_metric = y_cls_for_metric - offset1
                    if noise_label_for_metric is not None:
                        noise_label_for_metric = noise_label_for_metric - offset1

                prec = macro_precision_signal_only(pb, y_cls_for_metric, noise_label_for_metric)
                f1 = macro_f1_including_noise(pb, y_cls_for_metric)

                if y_det_for_metric is not None:
                    cls_mask = y_det_for_metric == 1
                    if cls_mask.any():
                        cls_tr_rec = macro_recall(pb[cls_mask], y_cls_for_metric[cls_mask])
                    else:
                        cls_tr_rec = 0.0
                elif noise_label_for_metric is not None:
                    cls_mask = y_cls_for_metric != noise_label_for_metric
                    if cls_mask.any():
                        cls_tr_rec = macro_recall(pb[cls_mask], y_cls_for_metric[cls_mask])
                    else:
                        cls_tr_rec = 0.0
                else:
                    cls_tr_rec = macro_recall(pb, y_cls_for_metric)

                det_rec = 0.0
                det_fa = 0.0
                if det_logits is not None and y_det_for_metric is not None:
                    det_pred = (det_logits >= 0).long().cpu()
                    det_rec = macro_recall(det_pred, y_det_for_metric)
                    det_fa = _false_alarm_rate(det_pred, y_det_for_metric)
                elif noise_label_for_metric is not None:
                    det_targets = (y_cls_for_metric != noise_label_for_metric).long()
                    det_pred = (pb != noise_label_for_metric).long()
                    det_rec = macro_recall(det_pred, det_targets)
                    det_fa = _false_alarm_rate(det_pred, det_targets)

                result_acc_tr[-1] = cls_tr_rec
                epoch_train_accs[-1] = cls_tr_rec

                epoch_precisions.append(prec)
                epoch_f1s.append(f1)
                epoch_det_recalls.append(det_rec)
                epoch_det_fas.append(det_fa)

                prog_bar.set_description(
                    "T{}| Ep: {}/{}| Loss: {}| Rec: {}| Prec: {}| F1: {}| DetRec: {}| DetFA: {}".format(
                        task_info["task"], ep + 1, args.n_epochs, round(loss, 3),
                        round(cls_tr_rec, 2), round(prec, 2), round(f1, 2), round(det_rec, 2), round(det_fa, 2)
                    )
                )

                # prog_bar.set_description(
                #     "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                #         task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                #         round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                #     )
                # )

            # Run validation at end of epoch (after last batch) so val scores reflect current task
            if (ep % args.val_rate) == 0:
                eval_start = time.time()
                log_state(
                    args.state_logging,
                    "Task {} Epoch {}/{}: running validation (end of epoch)".format(
                        current_task, ep + 1, args.n_epochs
                    ),
                )
                val_acc = evaluator(model, test_task_loaders, args)
                val_acc, val_prec, val_f1, val_det_acc, val_det_fa = _split_eval_output(val_acc)
                epoch_eval_time += time.time() - eval_start
                result_acc_val.append(val_acc)
                result_val_a.append(val_acc)
                if val_prec is not None:
                    result_val_prec.append(val_prec)
                if val_f1 is not None:
                    result_val_f1.append(val_f1)
                if val_det_acc is not None:
                    result_val_det_a.append(val_det_acc)
                    last_tr_det = sum(val_det_acc) / len(val_det_acc) if val_det_acc else None
                if val_det_fa is not None:
                    result_val_det_fa.append(val_det_fa)
                    last_tr_fa = sum(val_det_fa) / len(val_det_fa) if val_det_fa else None
                result_val_t.append(task_info["task"])
                if val_det_acc is not None:
                    print("---- Eval at Epoch {}: cls {} | det_recall {} | det_fa {} ----".format(
                        ep, val_acc, val_det_acc, val_det_fa
                    ))
                else:
                    print("---- Eval at Epoch {}: {} ----".format(ep, val_acc))

                # Store per-evaluation validation metrics for this task (current epoch).
                # Index into the evaluator outputs with the current task id where possible.
                current_task_idx = task_info["task"]
                if isinstance(val_acc, (list, tuple)) and current_task_idx < len(val_acc):
                    per_epoch_val_cls_rec.append(float(val_acc[current_task_idx]))
                elif not isinstance(val_acc, (list, tuple)):
                    per_epoch_val_cls_rec.append(float(val_acc))
                else:
                    per_epoch_val_cls_rec.append(float("nan"))

                if val_prec is not None:
                    if isinstance(val_prec, (list, tuple)) and current_task_idx < len(val_prec):
                        per_epoch_val_cls_prec.append(float(val_prec[current_task_idx]))
                    elif not isinstance(val_prec, (list, tuple)):
                        per_epoch_val_cls_prec.append(float(val_prec))
                    else:
                        per_epoch_val_cls_prec.append(float("nan"))
                else:
                    per_epoch_val_cls_prec.append(float("nan"))

                if val_f1 is not None:
                    if isinstance(val_f1, (list, tuple)) and current_task_idx < len(val_f1):
                        per_epoch_val_f1.append(float(val_f1[current_task_idx]))
                    elif not isinstance(val_f1, (list, tuple)):
                        per_epoch_val_f1.append(float(val_f1))
                    else:
                        per_epoch_val_f1.append(float("nan"))
                else:
                    per_epoch_val_f1.append(float("nan"))

                if val_det_acc is not None:
                    if isinstance(val_det_acc, (list, tuple)) and current_task_idx < len(val_det_acc):
                        per_epoch_val_det_rec.append(float(val_det_acc[current_task_idx]))
                    elif not isinstance(val_det_acc, (list, tuple)):
                        per_epoch_val_det_rec.append(float(val_det_acc))
                    else:
                        per_epoch_val_det_rec.append(float("nan"))
                else:
                    per_epoch_val_det_rec.append(float("nan"))

                if val_det_fa is not None:
                    if isinstance(val_det_fa, (list, tuple)) and current_task_idx < len(val_det_fa):
                        per_epoch_val_det_pfa.append(float(val_det_fa[current_task_idx]))
                    elif not isinstance(val_det_fa, (list, tuple)):
                        per_epoch_val_det_pfa.append(float(val_det_fa))
                    else:
                        per_epoch_val_det_pfa.append(float("nan"))
                else:
                    per_epoch_val_det_pfa.append(float("nan"))

            epoch_duration = time.time() - epoch_start_time
            epoch_train_time = max(epoch_duration - epoch_eval_time, 0.0)
            avg_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
            avg_cls_tr_rec = float(sum(epoch_train_accs) / len(epoch_train_accs)) if epoch_train_accs else float("nan")
            avg_prec = (
                float(sum(epoch_precisions) / len(epoch_precisions))
                if epoch_precisions
                else float("nan")
            )
            avg_f1 = float(sum(epoch_f1s) / len(epoch_f1s)) if epoch_f1s else float("nan")
            avg_det_rec = (
                float(sum(epoch_det_recalls) / len(epoch_det_recalls)) if epoch_det_recalls else float("nan")
            )
            avg_det_fa = (
                float(sum(epoch_det_fas) / len(epoch_det_fas)) if epoch_det_fas else float("nan")
            )

            # Track the last training metrics we saw (for summary logging).
            last_tr_cls_rec = avg_cls_tr_rec
            last_tr_cls_prec = avg_prec
            last_tr_cls_f1 = avg_f1

            # Persist per-epoch training metrics for this task.
            per_epoch_train_cls_rec.append(avg_cls_tr_rec)
            per_epoch_train_cls_prec.append(avg_prec)
            per_epoch_train_det_rec.append(avg_det_rec)
            per_epoch_train_det_pfa.append(avg_det_fa)
            per_epoch_train_f1.append(avg_f1)

            if not interactive_terminal:
                print(
                    "Task {} Epoch {}/{} | L {:.4f} | Train Acc {:.2f} | Prec {:.2f} | F1 {:.2f} | Det Rec {:.2f} | Det FA {:.2f} | Epoch Time {:.2f}s (Eval {:.2f}s, Train {:.2f}s)".format(
                        task_info["task"], ep + 1, args.n_epochs, avg_loss, avg_cls_tr_rec, avg_prec, avg_f1, avg_det_rec, avg_det_fa,
                        epoch_duration, epoch_eval_time, epoch_train_time
                    )
                )
                log_state(
                    args.state_logging,
                    "Task {} Epoch {}/{} complete: Prec {:.4f} F1 {:.4f} DetRec {:.4f} DetFA {:.4f} | {:.2f}s total ({:.2f}s eval/{:.2f}s train)".format(
                        current_task, ep + 1, args.n_epochs, avg_prec, avg_f1, avg_det_rec, avg_det_fa,
                        epoch_duration, epoch_eval_time, epoch_train_time
                    ),
                )
            if epoch_train_accs and epoch_eval_mode_recalls:
                avg_tr_recall = float(sum(epoch_train_accs) / len(epoch_train_accs))
                avg_eval_recall = float(sum(epoch_eval_mode_recalls) / len(epoch_eval_mode_recalls))
                print(
                    "Task {} Epoch {}/{} | Avg Train Recall {:.5f} | Avg Eval-Mode Recall {:.5f}".format(
                        task_info["task"], ep + 1, args.n_epochs, avg_tr_recall, avg_eval_recall
                    )
                )
        log_state(args.state_logging, "Task {}: running final validation.".format(current_task))
        val_acc = evaluator(model, test_task_loaders, args)
        val_acc, val_prec, val_f1, val_det_acc, val_det_fa = _split_eval_output(val_acc)
        result_val_a.append(val_acc)
        if val_prec is not None:
            result_val_prec.append(val_prec)
        if val_f1 is not None:
            result_val_f1.append(val_f1)
        if val_det_acc is not None:
            result_val_det_a.append(val_det_acc)
        if val_det_fa is not None:
            result_val_det_fa.append(val_det_fa)
        result_val_t.append(task_info["task"])

        losses = np.array(result_epoch_loss)
        result_acc_tr = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in result_acc_tr])
        result_acc_val = np.array(
            [x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in result_acc_val for x in sublist]
        )
        # Flatten validation F1 scores in the same order as result_acc_val, if available.
        if result_val_f1:
            result_val_f1_flat = np.array(
                [x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in result_val_f1 for x in sublist]
            )
        else:
            result_val_f1_flat = None

        logs_dir = os.path.join(args.log_dir, "metrics")
        os.makedirs(logs_dir, exist_ok=True)
        save_payload = {"losses": losses, "cls_tr_rec": result_acc_tr, "val_acc": result_acc_val}
        if result_val_f1_flat is not None:
            save_payload["val_f1"] = result_val_f1_flat
        # Optional: per-epoch training metrics for this task.
        if per_epoch_train_cls_rec:
            save_payload["train_cls_rec"] = np.asarray(per_epoch_train_cls_rec, dtype=float)
        if per_epoch_train_cls_prec:
            save_payload["train_cls_prec"] = np.asarray(per_epoch_train_cls_prec, dtype=float)
        if per_epoch_train_det_rec:
            save_payload["train_det_rec"] = np.asarray(per_epoch_train_det_rec, dtype=float)
        if per_epoch_train_det_pfa:
            save_payload["train_det_pfa"] = np.asarray(per_epoch_train_det_pfa, dtype=float)
        if per_epoch_train_f1:
            save_payload["train_f1"] = np.asarray(per_epoch_train_f1, dtype=float)

        # Optional: per-evaluation validation metrics for this task (one entry per eval/epoch).
        if per_epoch_val_cls_rec:
            save_payload["val_cls_rec"] = np.asarray(per_epoch_val_cls_rec, dtype=float)
        if per_epoch_val_cls_prec:
            save_payload["val_cls_prec"] = np.asarray(per_epoch_val_cls_prec, dtype=float)
        if per_epoch_val_det_rec:
            save_payload["val_det_rec"] = np.asarray(per_epoch_val_det_rec, dtype=float)
        if per_epoch_val_det_pfa:
            save_payload["val_det_pfa"] = np.asarray(per_epoch_val_det_pfa, dtype=float)
        if per_epoch_val_f1:
            save_payload["val_f1_per_epoch"] = np.asarray(per_epoch_val_f1, dtype=float)

        if result_val_det_a:
            save_payload["val_det_acc"] = np.array(result_val_det_a[-1])
        if result_val_det_fa:
            save_payload["val_det_fa"] = np.array(result_val_det_fa[-1])

        # Persist per-task metrics and a human-readable task order file.
        np.savez(os.path.join(logs_dir, "task" + str(task_i) + ".npz"), **save_payload)

        task_order_path = os.path.join(logs_dir, "task_order.txt")
        try:
            task_name = task_info.get("task_name", f"task{task_i}")
        except AttributeError:
            task_name = f"task{task_i}"
        with open(task_order_path, "a", encoding="utf-8") as f_task_order:
            f_task_order.write(str(task_name) + "\n")

        if args.calc_test_accuracy:
            test_acc = evaluator(model, test_task_loaders, args)
            test_acc, test_prec, test_f1, test_det_acc, test_det_fa = _split_eval_output(test_acc)
            result_test_a.append(test_acc)
            if test_det_acc is not None:
                result_test_det_a.append(test_det_acc)
            if test_det_fa is not None:
                result_test_det_fa.append(test_det_fa)
            result_test_t.append(task_info["task"])

        log_state(args.state_logging, "Completed task {} ({}/{})".format(current_task, task_i + 1, inc_loader.n_tasks))

    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))
    if result_val_det_a:
        print("Final Detection Results:- \n Total Detection: {} \n Individual Detection: {}".format(
            sum(result_val_det_a[-1]) / len(result_val_det_a[-1]), result_val_det_a[-1]
        ))
    if result_val_det_fa:
        print("Final Detection False Alarm:- \n Total False Alarm: {} \n Individual False Alarm: {}".format(
            sum(result_val_det_fa[-1]) / len(result_val_det_fa[-1]), result_val_det_fa[-1]
        ))

    def _mean(x):
        if x is None or (isinstance(x, (list, tuple)) and len(x) == 0):
            return None
        if isinstance(x, (list, tuple)):
            return sum(float(v) for v in x) / len(x)
        return float(x)

    if last_tr_cls_rec is not None or last_tr_cls_prec is not None or last_tr_cls_f1 is not None:
        tr_rec = float(last_tr_cls_rec) if last_tr_cls_rec is not None else None
        tr_prec = float(last_tr_cls_prec) if last_tr_cls_prec is not None else None
        tr_f1 = float(last_tr_cls_f1) if last_tr_cls_f1 is not None else None
        tr_det = last_tr_det
        tr_fa = last_tr_fa
        parts = []
        if tr_rec is not None:
            parts.append("cls_rec={:.4f}".format(tr_rec))
        if tr_prec is not None:
            parts.append("cls_prec={:.4f}".format(tr_prec))
        if tr_f1 is not None:
            parts.append("cls_f1={:.4f}".format(tr_f1))
        if tr_det is not None:
            parts.append("det={:.4f}".format(tr_det))
        if tr_fa is not None:
            parts.append("fa={:.4f}".format(tr_fa))
        if parts:
            print("SUMMARY_TR " + " ".join(parts))

    if result_val_a:
        te_rec = _mean(result_val_a[-1])
        te_prec = _mean(result_val_prec[-1]) if result_val_prec else None
        te_f1 = _mean(result_val_f1[-1]) if result_val_f1 else None
        te_det = _mean(result_val_det_a[-1]) if result_val_det_a else None
        te_fa = _mean(result_val_det_fa[-1]) if result_val_det_fa else None
        parts = ["cls_rec={:.4f}".format(te_rec)]
        if te_prec is not None:
            parts.append("cls_prec={:.4f}".format(te_prec))
        if te_f1 is not None:
            parts.append("cls_f1={:.4f}".format(te_f1))
        if te_det is not None:
            parts.append("det={:.4f}".format(te_det))
        if te_fa is not None:
            parts.append("fa={:.4f}".format(te_fa))
        print("SUMMARY_TE " + " ".join(parts))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))
        if result_test_det_a:
            print("Final Detection Results:- \n Total Detection: {} \n Individual Detection: {}".format(
                sum(result_test_det_a[-1]) / len(result_test_det_a[-1]), result_test_det_a[-1]
            ))
        if result_test_det_fa:
            print("Final Detection False Alarm:- \n Total False Alarm: {} \n Individual False Alarm: {}".format(
                sum(result_test_det_fa[-1]) / len(result_test_det_fa[-1]), result_test_det_fa[-1]
            ))


    time_end = time.time()
    time_spent = time_end - time_start

    def _pad_results(result_list: list[object], pad_value: float = 0.0) -> torch.Tensor:
        """Pad ragged per-task results into a dense 2D tensor.

        Args:
            result_list: Sequence of per-eval results, each being a list/array/tensor
                of task metrics or a scalar.
            pad_value: Value used to pad missing task entries.

        Returns:
            A 2D tensor of shape (num_evals, max_tasks).

        Usage:
            results = _pad_results([[0.1, 0.2], [0.3]])
        """
        if not result_list:
            return torch.empty((0, 0), dtype=torch.float)

        def _flatten_to_floats(value: object) -> list[float]:
            if isinstance(value, torch.Tensor):
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            if isinstance(value, np.ndarray):
                return [float(x) for x in value.flatten().tolist()]
            if isinstance(value, (list, tuple)):
                flattened: list[float] = []
                for item in value:
                    flattened.extend(_flatten_to_floats(item))
                return flattened
            return [float(value)]

        normalized_rows: list[torch.Tensor] = []
        for row in result_list:
            if row is None:
                row_tensor = torch.empty((0,), dtype=torch.float)
            else:
                row_tensor = torch.as_tensor(_flatten_to_floats(row), dtype=torch.float)
            normalized_rows.append(row_tensor)

        max_len = max(row.numel() for row in normalized_rows)
        padded = torch.full((len(normalized_rows), max_len), float(pad_value), dtype=torch.float)
        for row_idx, row_tensor in enumerate(normalized_rows):
            if row_tensor.numel() == 0:
                continue
            padded[row_idx, : row_tensor.numel()] = row_tensor
        return padded

    return (
        torch.Tensor(result_val_t),
        _pad_results(result_val_a),
        torch.Tensor(result_test_t),
        _pad_results(result_test_a),
        _pad_results(result_val_det_a),
        _pad_results(result_val_det_fa),
        _pad_results(result_test_det_a),
        _pad_results(result_test_det_fa),
        time_spent,
    )


def estimate_memory_buffer_size_bytes(model: torch.nn.Module) -> int:
    """Estimate total bytes used by replay/memory buffers in a model.

    This scans all modules for tensor attributes whose names suggest they are
    part of a replay or memory buffer (for example, attributes containing
    ``\"mem\"``) while excluding tensors already counted as parameters and
    avoiding double-counting shared storages.

    Args:
        model: Torch module whose memory/replay buffers will be inspected.

    Returns:
        Total number of bytes occupied by the matching tensors.

    Usage:
        buffer_bytes = estimate_memory_buffer_size_bytes(model)
    """
    parameter_data_ids = {id(parameter.data) for parameter in model.parameters()}
    seen_tensor_ids: set[int] = set()
    total_bytes = 0

    for module in model.modules():
        for attribute_name, value in vars(module).items():
            if not torch.is_tensor(value):
                continue
            if "mem" not in attribute_name.lower():
                continue
            tensor_data = value
            tensor_id = id(tensor_data)
            if tensor_id in seen_tensor_ids or tensor_id in parameter_data_ids:
                continue
            seen_tensor_ids.add(tensor_id)
            total_bytes += tensor_data.numel() * tensor_data.element_size()

    return total_bytes


def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')
    log_state(args.state_logging, "Saving results to {}".format(fname))

    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_gb = size_bytes / (1024 ** 3)
    buffer_bytes = estimate_memory_buffer_size_bytes(model)
    buffer_gb = buffer_bytes / (1024 ** 3)
    print("Model size: {:.4f} GB".format(size_gb))
    print("Memory buffer size: {:.4f} GB".format(buffer_gb))

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt')
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])
    one_liner += " # sizes: model_gb={:.4f} mem_gb={:.4f}".format(size_gb, buffer_gb)

    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    state_dict = model.state_dict()
    if getattr(args, "state_logging", False):
        def _tensor_storage_size(t):
            return t.element_size() * t.numel() if torch.is_tensor(t) else 0
        state_dict_bytes = sum(_tensor_storage_size(v) for v in state_dict.values())
        val_t_bytes = _tensor_storage_size(result_val_t)
        val_a_bytes = _tensor_storage_size(result_val_a)
        log_state(
            args.state_logging,
            "results.pt components (approx): state_dict {:.1f} MB, result_val_t {:.1f} KB, result_val_a {:.1f} KB".format(
                state_dict_bytes / (1024 * 1024), val_t_bytes / 1024, val_a_bytes / 1024
            ),
        )
    if hasattr(args, "get_samples_per_task"):
        try:
            delattr(args, "get_samples_per_task")
        except AttributeError:
            args.get_samples_per_task = None

    torch.save(
        (result_val_t, result_val_a, state_dict, val_stats, one_liner, args),
        fname + ".pt",
        pickle_protocol=4,
    )
    return val_stats, test_stats

def _default_main_config_chain() -> List[str]:
    chain: List[str] = []
    base_cfg = Path("configs/base.yaml")
    if base_cfg.exists():
        chain.append(str(base_cfg))
    legacy = Path("config_all.yaml")
    if legacy.exists():
        chain.append(str(legacy))
    return chain


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config fragment to apply (may be provided multiple times).",
    )
    config_parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory of YAML fragments to apply in alphabetical order.",
    )
    config_parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip loading YAML configs and rely solely on CLI arguments.",
    )
    config_cli, remaining = config_parser.parse_known_args()

    config_chain: List[str] = []
    if not config_cli.no_config:
        # Apply defaults first so explicit model configs override them.
        config_chain.extend(config_cli.config_dir)
        config_chain.extend(config_cli.config)
        if not config_chain:
            config_chain = _default_main_config_chain()

    base_args = file_parser.parse_args_from_yaml(config_chain or None)
    parser = file_parser.get_parser()
    args = parser.parse_args(remaining, namespace=base_args)

    # Scale learning rate based on batch size (reference batch size = 128).
    # This applies uniformly across all models that rely on args.lr.
    args.lr = misc_utils.scale_learning_rate_for_batch_size(args.lr, args.batch_size)
    print("Running model: ", args.model)
    log_state(
        args.state_logging,
        "Experiment '{}' starting with model '{}' (seed {})".format(args.expt_name, args.model, args.seed),
    )

    # initialize seeds
    misc_utils.init_seed(args.seed)

    # set up loader
    # 2 options: class_incremental and task_incremental
    # experiments in the paper only use task_incremental
    Loader = importlib.import_module('dataloaders.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()
    args.get_samples_per_task = getattr(loader, "get_samples_per_task", None)
    args.classes_per_task = getattr(loader, "classes_per_task", None)
    print("Classes per task:", args.classes_per_task)
    if args.classes_per_task is None or len(args.classes_per_task) == 0:
        args.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=args.nc_per_task_list if getattr(args, "nc_per_task_list", "") else args.nc_per_task,
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        print("Built classes_per_task:", args.classes_per_task)
    log_state(
        args.state_logging,
        "Loader '{}' ready: {} inputs, {} outputs, {} tasks".format(args.loader, n_inputs, n_outputs, n_tasks),
    )

    print("n_outputs:", n_outputs, "\tn_tasks:", n_tasks)

    # setup logging
    timestamp = misc_utils.get_date_time()
    config_name = Path(config_chain[-1]).stem if config_chain else None
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp, config_name)
    log_state(args.state_logging, "Logging to {}".format(args.log_dir))

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    # print(model)
    if args.cuda:
        try:
            model.cuda()
        except RuntimeError:
            pass
    print(args.cuda)
    print("Model device:", next(model.parameters()).device)
    log_state(args.state_logging, "Model initialized on device {}".format(next(model.parameters()).device))
    # run model on loader
    if args.model == "iid2":
        # oracle baseline with all task data shown at same time
        log_state(args.state_logging, "Invoking iid life experience flow")
        result_val_t, result_val_a, result_test_t, result_test_a, _, _, _, _, spent_time = (
            life_experience_iid(model, loader, args)
        )
    else:
        # for all the CL baselines
        log_state(args.state_logging, "Invoking continual life experience flow")
        result_val_t, result_val_a, result_test_t, result_test_a, _, _, _, _, spent_time = (
            life_experience(model, loader, args)
        )

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)
        log_state(args.state_logging, "Results saved; total runtime {:.2f}s".format(spent_time))


if __name__ == "__main__":
    print("New Experiment Starting...")
    main()
