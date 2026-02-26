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
from dataloaders.iq_data_loader import ensure_iq_two_channel
from utils.training_metrics import macro_recall

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
        return 0.0
    neg_targets = targets[neg_mask]
    neg_preds = preds[neg_mask]
    fp = (neg_preds == 1).sum().item()
    tn = (neg_targets == 0).sum().item() - fp
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0

def _noise_label_for_task(args, task_idx: int) -> int | None:
    class_counts = getattr(args, "classes_per_task", None)
    if class_counts is None:
        return None
    _, offset2 = misc_utils.compute_offsets(task_idx, class_counts)
    return offset2 - 1

def eval_class_tasks(model, tasks, args):

    model.eval()
    result = []
    for t, task_loader in enumerate(tasks):
        correct = 0.0
        total = 0.0
        noise_label = _noise_label_for_task(args, t)

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
    is_iq = getattr(args, 'dataset', '').lower() == 'iq'
    class_counts = getattr(args, "classes_per_task", None)
    batch_size = getattr(args, 'eval_batch_size', 256)

    if specific_task is not None:
        tasks = [tasks[specific_task]]
        batch_size = 256

    def _task_is_dataset(task: object) -> bool:
        """Return True when the task is a dataset tuple with array-like samples.

        Args:
            task: Task payload produced by the incremental loader.

        Returns:
            True if the task is a 3-tuple of (meta, x, y).
        """
        if not isinstance(task, (list, tuple)) or len(task) != 3:
            return False
        return isinstance(task[1], np.ndarray) or torch.is_tensor(task[1])
    
    det_results = []
    det_fa_results = []
    det_metrics_active = False
    for i, task in enumerate(tasks):
        t = i
        if not _task_is_dataset(task):
            recalls = []
            det_recalls = []
            det_false_alarms = []
            noise_label = _noise_label_for_task(args, t)
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
                if yb_det is not None and not torch.is_tensor(yb_det):
                    yb_det = torch.as_tensor(yb_det)

                logits = model(xb, t) if args.model != "anml" else model(xb, fast_weights=None)
                pb = torch.argmax(logits, dim=1).cpu()
                yb_cls_cpu = yb_cls.detach().cpu()
                yb_det_cpu = yb_det.detach().cpu() if yb_det is not None else None
                if yb_det_cpu is not None:
                    cls_mask = yb_det_cpu == 1
                    if cls_mask.any():
                        recalls.append(macro_recall(pb[cls_mask], yb_cls_cpu[cls_mask]))
                elif noise_label is not None:
                    cls_mask = yb_cls_cpu != noise_label
                    if cls_mask.any():
                        recalls.append(macro_recall(pb[cls_mask], yb_cls_cpu[cls_mask]))
                else:
                    recalls.append(macro_recall(pb, yb_cls_cpu))

                if yb_det_cpu is not None:
                    det_logits = _get_det_logits(model, xb, t)
                    if det_logits is not None:
                        det_pred = (det_logits >= 0).long().cpu()
                        det_recalls.append(macro_recall(det_pred, yb_det_cpu))
                        det_false_alarms.append(_false_alarm_rate(det_pred, yb_det_cpu))
                elif noise_label is not None:
                    yb_unique = torch.unique(yb_cls_cpu)
                    pred_unique = torch.unique(pb)
                    noise_present = (yb_unique == noise_label).any().item()
                    det_targets = (yb_cls_cpu != noise_label).long()
                    det_pred = (pb != noise_label).long()
                    # print(
                    #     "DEBUG det: noise_label={}, present={}, yb_unique={}, pb_unique={}, "
                    #     "pos_target_rate={:.3f}, pos_pred_rate={:.3f}".format(
                    #         noise_label,
                    #         noise_present,
                    #         yb_unique.tolist(),
                    #         pred_unique.tolist(),
                    #         det_targets.float().mean().item(),
                    #         det_pred.float().mean().item(),
                    #     )
                    # )
                    # det_recalls.append(macro_recall(det_pred, det_targets))
                    det_false_alarms.append(_false_alarm_rate(det_pred, det_targets))

            results.append(sum(recalls) / len(recalls) if recalls else 0.0)
            if det_recalls:
                det_results.append(sum(det_recalls) / len(det_recalls))
                det_fa_results.append(sum(det_false_alarms) / len(det_false_alarms))
                det_metrics_active = True
            else:
                det_results.append(0.0)
                det_fa_results.append(0.0)
            continue
        x_data = task[1]
        y_cls_raw, y_det_raw = _split_labels(task[2])
        y = torch.as_tensor(y_cls_raw, dtype=torch.long)
        y_det = None
        if y_det_raw is not None:
            y_det = torch.as_tensor(y_det_raw, dtype=torch.long)
        # if y_det is None: print("Warning: y_det is None for Task {}, defaulting to all ones (all samples treated as CLS).".format(t))
        noise_label = _noise_label_for_task(args, t)
        if 'ucl' in args.model:
            offset1, offset2 = misc_utils.compute_offsets(
                t, class_counts if class_counts is not None else args.nc_per_task
            )
            if y_det is not None:
                y = y.clone()
                mask = y_det == 1
                if mask.any():
                    y[mask] = y[mask] - offset1
            else:
                y = y - offset1  # make labels start from 0 for each task
                if noise_label is not None:
                    noise_label = noise_label - offset1

        if isinstance(x_data, torch.Tensor):
            x_data_cpu = x_data.detach().cpu()
            if is_iq:
                # print("Original test shape:", x_data_cpu.shape)
                x_np = ensure_iq_two_channel(x_data_cpu.numpy())
                x = torch.from_numpy(x_np)
                # print("Converted IQ data to 2-channel format, new shape:", x.shape)
            else:
                x = x_data_cpu.float()
        else:
            if is_iq:
                # print("Original test shape:", x_data.shape)
                x_np = ensure_iq_two_channel(x_data)
                x = torch.from_numpy(x_np)
                # print("Converted IQ data to 2-channel format, new shape:", x.shape)
            else:
                x = torch.from_numpy(np.asarray(x_data, dtype=np.float32))

        x = x.float()

        recalls = []
        det_recalls = []
        det_false_alarms = []
        N = x.size(0)
        # print(f"Evaluating Task {t}: {N} samples, batch size {batch_size}, noise label {noise_label}")
        epistemic_uncertainties = []
        eh = []
        h_preds = []
        for b_from in range(0, N, batch_size):
            b_to = min(b_from + batch_size, N)
            xb = x[b_from:b_to].to(device)
            if getattr(args, 'arch', '').lower() == 'linear':
                xb = xb.view(xb.size(0), -1)
                
            yb = y[b_from:b_to].to(device)
            yb_det = None
            if y_det is not None:
                yb_det = y_det[b_from:b_to].to(device)

            logits = model(xb, t) if args.model != 'anml' else model(xb, fast_weights=None)
            pb = torch.argmax(logits, dim=1)
            # correct += (pb == yb).sum().item()
            if yb_det is not None:
                cls_mask = yb_det == 1
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask].cpu(), yb[cls_mask].cpu()))
            elif noise_label is not None:
                cls_mask = yb != noise_label
                if cls_mask.any():
                    recalls.append(macro_recall(pb[cls_mask].cpu(), yb[cls_mask].cpu()))
            else:
                recalls.append(macro_recall(pb.cpu(), yb.cpu()))

            if yb_det is not None:
                det_logits = _get_det_logits(model, xb, t)
                if det_logits is not None:
                    det_pred = (det_logits >= 0).long()
                    det_recalls.append(macro_recall(det_pred.cpu(), yb_det.cpu()))
                    det_false_alarms.append(_false_alarm_rate(det_pred, yb_det))
            elif noise_label is not None:
                det_targets = (yb != noise_label).long()
                det_pred = (pb != noise_label).long()
                det_recalls.append(macro_recall(det_pred.cpu(), det_targets.cpu()))
                det_false_alarms.append(_false_alarm_rate(det_pred, det_targets))
            if eval_epistemic and 'ucl' in args.model:
                p_mean, H_pred, EH, MI = model.mc_epistemic_classification(xb, t, S=30)
                epistemic_uncertainties.append(MI.mean().cpu().item())
                eh.append(EH.mean().cpu().item())
                h_preds.append(H_pred.mean().cpu().item())
        
        if eval_epistemic and 'ucl' in args.model:
            print("---- Epistemic Uncertainty Analysis for Task {} ----".format(i))
            print("H_pred min: {}, max: {}, mean: {}".format(min(h_preds), max(h_preds), sum(h_preds)/len(h_preds)))
            print("EH min: {}, max: {}, mean: {}".format(min(eh), max(eh), sum(eh)/len(eh)))
            print("EU min: {}, max: {}, mean: {}".format(min(epistemic_uncertainties), max(epistemic_uncertainties), sum(epistemic_uncertainties)/len(epistemic_uncertainties)))
            average_eh = 100* sum(eh)/len(eh)
            current_nc = misc_utils.task_class_count(class_counts, t) if class_counts is not None else args.nc_per_task
            norm_avg_eh = average_eh / np.log(current_nc)
            average_h_pred = 100* sum(h_preds)/len(h_preds)
            norm_avg_h_pred = average_h_pred / np.log(current_nc)
            average_eu = 100* sum(epistemic_uncertainties)/len(epistemic_uncertainties)
            norm_avg_eu = average_eu / np.log(current_nc)
            print("Task: {} | Epistemic: {} | Aleatoric: {} | Total: {} | F_eu | {}".format(
                i, round(norm_avg_eu, 2), round(norm_avg_eh, 2), round(norm_avg_h_pred, 2), round(norm_avg_eu/norm_avg_h_pred, 2)))

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
    # print("WARNING: Using training data for validation. This is for debugging purposes only!!")
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
            epoch_eval_mode_recalls = []
            epoch_start_time = time.time()
            epoch_eval_time = 0.0
            log_state(
                args.state_logging,
                "Task {} Epoch {}/{}: entering train loop".format(current_task, ep + 1, args.n_epochs),
            )

            prog_bar = tqdm(train_loader, disable=not interactive_terminal)
            for (i, (x, y)) in enumerate(prog_bar):

                if ((ep % args.val_rate) == 0) and ((i % 3125 == 0)):
                    eval_start = time.time()
                    log_state(
                        args.state_logging,
                        "Task {} Epoch {}/{} Iter {}: running validation".format(
                            current_task, ep + 1, args.n_epochs, i
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
                #         model(v_x, task_info["task"])
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

                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | Loss: {} | Acc: Task_avg: {} Tr: {} Val: {} ".format(
                        task_info["task"], ep+1, args.n_epochs, round(loss, 3),
                        round(sum(result_val_a[-1])/len(result_val_a[-1]), 5), round(tr_acc, 5), round(result_val_a[-1][task_info["task"]], 5)
                    )
                )

                # prog_bar.set_description(
                #     "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                #         task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                #         round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                #     )
                # )

            if not interactive_terminal:
                epoch_duration = time.time() - epoch_start_time
                epoch_train_time = max(epoch_duration - epoch_eval_time, 0.0)
                avg_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
                avg_tr_acc = float(sum(epoch_train_accs) / len(epoch_train_accs)) if epoch_train_accs else float("nan")
                latest_val = (
                    result_acc_val[-1][task_info["task"]]
                    if result_acc_val and task_info["task"] < len(result_acc_val[-1])
                    else float("nan")
                )
                print(
                    "Task {} Epoch {}/{} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Epoch Time {:.2f}s (Eval {:.2f}s, Train {:.2f}s)".format(
                        task_info["task"], ep + 1, args.n_epochs, avg_loss, avg_tr_acc, latest_val,
                        epoch_duration, epoch_eval_time, epoch_train_time
                    )
                )
                log_state(
                    args.state_logging,
                    "Task {} Epoch {}/{} complete: {:.2f}s total ({:.2f}s eval/{:.2f}s train)".format(
                        current_task, ep + 1, args.n_epochs, epoch_duration, epoch_eval_time, epoch_train_time
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
        evaluator(model, train_task_loaders, args, eval_epistemic=False)
        val_acc = evaluator(model, train_task_loaders, args)
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

    def _pad_results(result_list: list[list[float]], pad_value: float = 0.0) -> torch.Tensor:
        if not result_list:
            return torch.empty((0, 0), dtype=torch.float)
        max_len = max(len(row) for row in result_list)
        padded = torch.full((len(result_list), max_len), float(pad_value), dtype=torch.float)
        for row_idx, row in enumerate(result_list):
            if not row:
                continue
            padded[row_idx, : len(row)] = torch.as_tensor(row, dtype=torch.float)
        return padded

    return (
        torch.Tensor(result_val_t),
        _pad_results(result_val_a),
        torch.Tensor(result_test_t),
        _pad_results(result_test_a),
        torch.Tensor(result_val_det_a),
        torch.Tensor(result_val_det_fa),
        torch.Tensor(result_test_det_a),
        torch.Tensor(result_test_det_fa),
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
    # NOTE: avoid serializing the full ``args`` namespace because it may contain
    # loader methods and dataset references, which can make the pickle several
    # gigabytes in size. The stringified config in ``one_liner`` already
    # captures the experiment setup for later inspection.
    torch.save(
        (result_val_t, result_val_a, model.state_dict(), val_stats, one_liner),
        fname + ".pt",
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
