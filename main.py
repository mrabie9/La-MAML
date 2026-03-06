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
    if isinstance(output, (tuple, list)):
        if len(output) == 3:
            return output[0], output[1], output[2]
        if len(output) == 2:
            return output[0], output[1], None
    return output, None, None

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
    return offset2 - 1


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
            if yb_det_cpu is not None:
                cls_mask = yb_det_cpu == 1
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask], yb_cls_for_metrics[cls_mask]))
            elif noise_label_for_metrics is not None:
                cls_mask = yb_cls_for_metrics != noise_label_for_metrics
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask], yb_cls_for_metrics[cls_mask]))
            else:
                recalls.append(macro_recall(pb, yb_cls_for_metrics))

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
        if det_recalls:
            det_results.append(sum(det_recalls) / len(det_recalls))
            det_fa_results.append(sum(det_false_alarms) / len(det_false_alarms))
            det_metrics_active = True
        else:
            det_results.append(0.0)
            det_fa_results.append(0.0)

    if det_metrics_active:
        return results, det_results, det_fa_results
    return results, None

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []
    result_val_det_a = []
    result_test_det_a = []
    result_val_det_fa = []
    result_test_det_fa = []

    result_val_t = []
    result_test_t = []

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

                loss, tr_acc = model.observe(Variable(v_x), v_y, task_info["task"])
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
                #             "DEBUG batch0: tr_acc {:.5f} | eval_mode_recall {:.5f} | noise_label {}".format(
                #                 float(tr_acc), float(debug_eval_recall), debug_noise_label
                #             )
                #         )
                # model.train()
                # logits = model(x, task_i) if args.model != 'anml' else model(x, task_i, fast_weights=None)
                # pb = torch.argmax(logits, dim=1)
                # correct += (pb == y).sum().item()
                # tr_acc = correct / x.size(0)
                result_acc_tr.append(tr_acc)
                result_epoch_loss.append(loss)
                epoch_losses.append(loss)
                epoch_train_accs.append(tr_acc)

                # Batch-level precision (signal only) and F1 (all classes incl. noise) for progress bar
                noise_label = _noise_label_for_task(args, task_info["task"])
                model.eval()
                with torch.no_grad():
                    logits = (
                        model(v_x, task_info["task"])
                        if args.model != "anml"
                        else model(v_x, fast_weights=None)
                    )
                    pb = torch.argmax(logits, dim=1)
                model.train()
                y_cls_for_metric = y_cls if torch.is_tensor(y_cls) else torch.as_tensor(y_cls)
                prec = macro_precision_signal_only(pb, y_cls_for_metric, noise_label)
                f1 = macro_f1_including_noise(pb, y_cls_for_metric)
                epoch_precisions.append(prec)
                epoch_f1s.append(f1)
                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | Loss: {} | Tr: {} | Prec: {} | F1: {} ".format(
                        task_info["task"], ep + 1, args.n_epochs, round(loss, 3),
                        round(tr_acc, 5), round(prec, 5), round(f1, 5),
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
                val_acc, val_det_acc, val_det_fa = _split_eval_output(val_acc)
                epoch_eval_time += time.time() - eval_start
                result_acc_val.append(val_acc)
                result_val_a.append(val_acc)
                if val_det_acc is not None:
                    result_val_det_a.append(val_det_acc)
                if val_det_fa is not None:
                    result_val_det_fa.append(val_det_fa)
                result_val_t.append(task_info["task"])
                if val_det_acc is not None:
                    print("---- Eval at Epoch {}: cls {} | det_recall {} | det_fa {} ----".format(
                        ep, val_acc, val_det_acc, val_det_fa
                    ))
                else:
                    print("---- Eval at Epoch {}: {} ----".format(ep, val_acc))

            if not interactive_terminal:
                epoch_duration = time.time() - epoch_start_time
                epoch_train_time = max(epoch_duration - epoch_eval_time, 0.0)
                avg_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
                avg_tr_acc = float(sum(epoch_train_accs) / len(epoch_train_accs)) if epoch_train_accs else float("nan")
                avg_prec = (
                    float(sum(epoch_precisions) / len(epoch_precisions))
                    if epoch_precisions
                    else float("nan")
                )
                avg_f1 = (
                    float(sum(epoch_f1s) / len(epoch_f1s)) if epoch_f1s else float("nan")
                )
                print(
                    "Task {} Epoch {}/{} | Loss {:.4f} | Train Acc {:.4f} | Prec {:.4f} | F1 {:.4f} | Epoch Time {:.2f}s (Eval {:.2f}s, Train {:.2f}s)".format(
                        task_info["task"], ep + 1, args.n_epochs, avg_loss, avg_tr_acc, avg_prec, avg_f1,
                        epoch_duration, epoch_eval_time, epoch_train_time
                    )
                )
                log_state(
                    args.state_logging,
                    "Task {} Epoch {}/{} complete: Prec {:.4f} F1 {:.4f} | {:.2f}s total ({:.2f}s eval/{:.2f}s train)".format(
                        current_task, ep + 1, args.n_epochs, avg_prec, avg_f1,
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
        val_acc, val_det_acc, val_det_fa = _split_eval_output(val_acc)
        result_val_a.append(val_acc)
        if val_det_acc is not None:
            result_val_det_a.append(val_det_acc)
        if val_det_fa is not None:
            result_val_det_fa.append(val_det_fa)
        result_val_t.append(task_info["task"])

        losses = np.array(result_epoch_loss)
        # print(epoch_accuracies)
        result_acc_tr = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in result_acc_tr])
        # print(epoch_accuracies)
        result_acc_val = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in result_acc_val for x in sublist])
        logs_dir = os.path.join(args.log_dir, "metrics")
        os.makedirs(logs_dir, exist_ok=True)
        save_payload = {"losses": losses, "tr_acc": result_acc_tr, "val_acc": result_acc_val}
        if result_val_det_a:
            save_payload["val_det_acc"] = np.array(result_val_det_a[-1])
        if result_val_det_fa:
            save_payload["val_det_fa"] = np.array(result_val_det_fa[-1])
        np.savez(os.path.join(logs_dir, "task" + str(task_i)+".npz"), **save_payload) 

        if args.calc_test_accuracy:
            test_acc = evaluator(model, test_task_loaders, args)
            test_acc, test_det_acc, test_det_fa = _split_eval_output(test_acc)
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

def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')
    log_state(args.state_logging, "Saving results to {}".format(fname))

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt')
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])

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
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp)
    log_state(args.state_logging, "Logging to {}".format(args.log_dir))

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    # print(model)
    if args.cuda:
        try:
            model.cuda()            
        except:
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
