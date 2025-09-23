# TODO: combine resnet181d and lps dataloader with this repo
# TODO: make sure config options make sense
# TODO: run resnet181d with La-MAML on lora dataset

import importlib
import datetime
import argparse
import time
import os
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

def eval_tasks(model, tasks, args, batch_size=64):
    model.eval()
    device = torch.device('cuda' if getattr(args, 'cuda', False) and torch.cuda.is_available() else 'cpu')
    results = []
    is_iq = getattr(args, 'dataset', '').lower() == 'iq'

    for i, task in enumerate(tasks):
        t = i
        x_data = task[1]
        y = torch.as_tensor(task[2], dtype=torch.long)

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

        correct = 0
        N = x.size(0)

        for b_from in range(0, N, batch_size):
            b_to = min(b_from + batch_size, N)
            xb = x[b_from:b_to].to(device)
            if getattr(args, 'arch', '').lower() == 'linear':
                xb = xb.view(xb.size(0), -1)
            yb = y[b_from:b_to].to(device)

            logits = model(xb, t)
            pb = torch.argmax(logits, dim=1)
            correct += (pb == yb).sum().item()

        results.append(correct / N)

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

    for task_i in range(inc_loader.n_tasks):
        task_info, train_loader, _, _ = inc_loader.new_task()
        for ep in range(args.n_epochs):

            model.real_epoch = ep

            prog_bar = tqdm(train_loader)
            for (i, (x, y)) in enumerate(prog_bar):

                if((i % args.log_every) == 0):
                    result_val_a.append(evaluator(model, val_tasks, args))
                    result_val_t.append(task_info["task"])

                v_x = x
                v_y = y
                if args.arch == 'linear':
                    v_x = x.view(x.size(0), -1)
                if args.cuda:
                    v_x = v_x.cuda()
                    v_y = v_y.cuda()

                model.train()

                loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])

                prog_bar.set_description(
                    "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                        task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                        round(sum(result_val_a[-1])/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]], 5)
                    )
                )

                # prog_bar.set_description(
                #     "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                #         task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                #         round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                #     )
                # )

        result_val_a.append(evaluator(model, val_tasks, args))
        result_val_t.append(task_info["task"])

        if args.calc_test_accuracy:
            result_test_a.append(evaluator(model, test_tasks, args))
            result_test_t.append(task_info["task"])


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
    args = file_parser.parse_args_from_yaml(yaml_file)
    # parser = file_parser.get_parser()
    # args = parser.parse_args()

    # initialize seeds
    misc_utils.init_seed(args.seed)

    # set up loader
    # 2 options: class_incremental and task_incremental
    # experiments in the paper only use task_incremental
    Loader = importlib.import_module('dataloaders.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    # setup logging
    timestamp = misc_utils.get_date_time()
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        try:
            model.net.cuda()            
        except:
            pass 

    # run model on loader
    if args.model == "iid2":
        # oracle baseline with all task data shown at same time
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience_iid(
            model, loader, args)
    else:
        # for all the CL baselines
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
            model, loader, args)

        # save results in files or print on terminal
        save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)


if __name__ == "__main__":
    main()
