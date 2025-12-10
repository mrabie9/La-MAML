# TODO: hyperparameter tuner

import importlib
import datetime
import argparse
import time
import os
import sys
import ipdb
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable

import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from main_multi_task import life_experience_iid, eval_iid_tasks
from dataloaders.iq_data_loader import ensure_iq_two_channel
from utils.training_metrics import macro_recall

def log_state(enabled, message):
    """Print a timestamped state message when state logging is enabled."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[STATE {}] {}".format(timestamp, message))

def eval_class_tasks(model, tasks, args):

    model.eval()
    result = []
    for t, task_loader in enumerate(tasks):
        correct = 0.0

        for (i, (x, y)) in enumerate(task_loader):
            if args.cuda:
                x = x.cuda()
            _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
            correct += (p == y).float().sum().item()

        result.append(correct / len(task_loader.dataset))
    return result

def eval_tasks(model, tasks, args, specific_task=None, eval_epistemic = False):
    model.eval()
    device = torch.device('cuda' if getattr(args, 'cuda', False) and torch.cuda.is_available() else 'cpu')
    results = []
    is_iq = getattr(args, 'dataset', '').lower() == 'iq'
    batch_size = getattr(args, 'eval_batch_size', 64)

    if specific_task is not None:
        tasks = [tasks[specific_task]]
        batch_size = 64
    
    for i, task in enumerate(tasks):
        t = i
        x_data = task[1]
        y = torch.as_tensor(task[2], dtype=torch.long)
        if 'ucl' in args.model:
            offset1, offset2 = misc_utils.compute_offsets(t, args.nc_per_task)
            y = y - offset1  # make labels start from 0 for each task

        if isinstance(x_data, torch.Tensor):
            x_data_cpu = x_data.detach().cpu()
            if is_iq:
                x_np = ensure_iq_two_channel(x_data_cpu.numpy())
                x = torch.from_numpy(x_np)
            else:
                x = x_data_cpu.float()
        else:
            if is_iq:
                x_np = ensure_iq_two_channel(x_data)
                x = torch.from_numpy(x_np)
            else:
                x = torch.from_numpy(np.asarray(x_data, dtype=np.float32))

        x = x.float()

        recalls = []
        N = x.size(0)
        epistemic_uncertainties = []
        eh = []
        h_preds = []
        for b_from in range(0, N, batch_size):
            b_to = min(b_from + batch_size, N)
            xb = x[b_from:b_to].to(device)
            if getattr(args, 'arch', '').lower() == 'linear':
                xb = xb.view(xb.size(0), -1)
                
            yb = y[b_from:b_to].to(device)

            logits = model(xb, t) if args.model != 'anml' else model(xb, fast_weights=None)
            pb = torch.argmax(logits, dim=1)
            # correct += (pb == yb).sum().item()
            recalls.append(macro_recall(pb.cpu(), yb.cpu()))
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
            norm_avg_eh = average_eh / np.log(args.nc_per_task)
            average_h_pred = 100* sum(h_preds)/len(h_preds)
            norm_avg_h_pred = average_h_pred / np.log(args.nc_per_task)
            average_eu = 100* sum(epistemic_uncertainties)/len(epistemic_uncertainties)
            norm_avg_eu = average_eu / np.log(args.nc_per_task)
            print("Task: {} | Epistemic: {} | Aleatoric: {} | Total: {} | F_eu | {}".format(
                i, round(norm_avg_eu, 2), round(norm_avg_eh, 2), round(norm_avg_h_pred, 2), round(norm_avg_eu/norm_avg_h_pred, 2)))

        results.append(sum(recalls) / len(recalls))

    return results
    
# def eval_tasks(model, tasks, args):

#     model.eval()
#     result = []
#     for i, task in enumerate(tasks):
#         t = i
#         x = task[1] # (8400, 4096)
#         y = task[2] # (8400, )
#         rt = 0
        
#         x = torch.tensor(x)
#         y = torch.tensor(y)
#         eval_bs = 256 #x.size(0)

#         for b_from in range(0, x.size(0), eval_bs):
#             b_to = min(b_from + eval_bs, x.size(0) - 1)
#             if b_from == b_to:
#                 xb = x[b_from].view(1, -1)
#                 yb = torch.LongTensor([y[b_to]]).view(1, -1)
#             else:
#                 xb = x[b_from:b_to]
#                 yb = y[b_from:b_to]
#             if args.cuda:
#                 xb = xb.cuda()
#             _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
#             rt += (pb == yb).float().sum()

#         result.append(rt / x.size(0))

#     return result

def life_experience(model, inc_loader, args):
    result_val_a = []
    result_test_a = []

    result_val_t = []
    result_test_t = []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")
    
    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    interactive_terminal = False # sys.stdout.isatty()
    log_state(args.state_logging, "Life experience start: {} tasks queued".format(inc_loader.n_tasks))

    for task_i in range(inc_loader.n_tasks):
        result_epoch_loss = []
        result_acc_val = []
        result_acc_tr = []
        task_info, train_loader, _, _ = inc_loader.new_task()
        current_task = task_info["task"]
        log_state(
            args.state_logging,
            "Starting task {} ({}/{})".format(current_task, task_i + 1, inc_loader.n_tasks),
        )
        for ep in range(args.n_epochs):
            model.real_epoch = ep
            epoch_losses = []
            epoch_train_accs = []
            epoch_start_time = time.time()
            epoch_eval_time = 0.0
            log_state(
                args.state_logging,
                "Task {} Epoch {}/{}: entering train loop".format(current_task, ep + 1, args.n_epochs),
            )

            prog_bar = tqdm(train_loader, disable=not interactive_terminal)
            for (i, (x, y)) in enumerate(prog_bar):

                if((ep % args.val_rate) == 0) and ((i % 3125 == 0)):
                    eval_start = time.time()
                    log_state(
                        args.state_logging,
                        "Task {} Epoch {}/{} Iter {}: running validation".format(
                            current_task, ep + 1, args.n_epochs, i
                        ),
                    )
                    val_acc = evaluator(model, val_tasks, args)
                    epoch_eval_time += time.time() - eval_start
                    result_acc_val.append(val_acc)
                    result_val_a.append(val_acc)
                    result_val_t.append(task_info["task"])
                    print("---- Eval at Epoch {}: {} ----".format(ep, val_acc))

                v_x = x
                v_y = y
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()
                model.train()

                loss, tr_acc = model.observe(Variable(v_x), Variable(v_y), task_info["task"])
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
        log_state(args.state_logging, "Task {}: running final validation.".format(current_task))
        evaluator(model, val_tasks, args, eval_epistemic=False)
        result_val_a.append(evaluator(model, val_tasks, args))
        result_val_t.append(task_info["task"])

        losses = np.array(result_epoch_loss)
        # print(epoch_accuracies)
        result_acc_tr = np.array([x.cpu().item() if torch.is_tensor(x) else x for x in result_acc_tr])
        # print(epoch_accuracies)
        result_acc_val = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for sublist in result_acc_val for x in sublist])
        logs_dir = os.path.join(args.log_dir, "metrics")
        os.makedirs(logs_dir, exist_ok=True)
        np.savez(os.path.join(logs_dir, "task" + str(task_i)+".npz"), losses=losses, tr_acc=result_acc_tr, val_acc=result_acc_val) 

        if args.calc_test_accuracy:
            result_test_a.append(evaluator(model, test_tasks, args))
            result_test_t.append(task_info["task"])

        log_state(args.state_logging, "Completed task {} ({}/{})".format(current_task, task_i + 1, inc_loader.n_tasks))

    print("####Final Validation Accuracy####")
    print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

    if args.calc_test_accuracy:
        print("####Final Test Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))


    time_end = time.time()
    time_spent = time_end - time_start
    return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent

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
    torch.save((result_val_t, result_val_a, model.state_dict(),
                val_stats, one_liner, args), fname + '.pt')
    return val_stats, test_stats

def main():
    base_path = ''
    yaml_file = 'config_all.yaml'
    # args = file_parser.parse_args_from_yaml(yaml_file)
    parser = file_parser.get_parser()
    args = parser.parse_args()
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
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience_iid(
            model, loader, args)
    else:
        # for all the CL baselines
        log_state(args.state_logging, "Invoking continual life experience flow")
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
            model, loader, args)

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)
        log_state(args.state_logging, "Results saved; total runtime {:.2f}s".format(spent_time))


if __name__ == "__main__":
    print("New Experiment Starting...")
    main()
