import torch
from dataclasses import dataclass

import numpy as np
import random

import sys
from utils.training_metrics import macro_recall
from model.resnet1d import ResNet1D
from utils import misc_utils

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("once")

"""
Multi task
    big batch size, set increment 100 so that it is treated as 1 task with all classes in the dataset
    inference time for acc eval, use offsets
"""


@dataclass
class IidConfig:
    arch: str = "resnet1d"
    n_layers: int = 2
    n_hiddens: int = 100
    dataset: str = "tinyimagenet"
    lr: float = 1e-3
    cuda: bool = True

    @staticmethod
    def from_args(args: object) -> "IidConfig":
        cfg = IidConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = IidConfig.from_args(args)
        self.nt = n_tasks

        self.n_feat = n_outputs
        self.n_classes = n_outputs

        arch = self.cfg.arch
        if arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=self.cfg.lr)

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()

        self.gpu = self.cfg.cuda
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.n_outputs = n_outputs

    def compute_offsets(self, task):
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def take_multitask_loss(self, bt, logits, y):
        loss = 0.0
        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def forward(self, x, t):                                  
                                                
        output = self.net.forward(x)

        # make sure we predict classes within the current task
        if torch.unique(t).shape[0] == 1:
            offset1, offset2 = self.compute_offsets(t[0].item())
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        else:
            for i in range(len(t)):
                offset1, offset2 = self.compute_offsets(t[i])
                if offset1 > 0:
                    output[i, :offset1].data.fill_(-10e10)
                if offset2 < self.n_outputs:
                    output[i, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):
        self.net.train()

        self.net.zero_grad()
        logits = self.net.forward(x)
        loss = self.take_multitask_loss(t, logits, y) 
        with torch.no_grad():
            batch_preds = []
            batch_targets = []
            for i, ti in enumerate(t):
                offset1, offset2 = self.compute_offsets(ti)
                preds = torch.argmax(logits[i, offset1:offset2], dim=0)
                target = y[i] - offset1
                batch_preds.append(preds.detach().cpu())
                batch_targets.append(target.detach().cpu())
            if batch_preds:
                stacked_preds = torch.stack(batch_preds).view(-1)
                stacked_targets = torch.stack(batch_targets).view(-1)
                tr_acc = macro_recall(stacked_preds, stacked_targets)
            else:
                tr_acc = 0.0
        loss.backward()
        self.opt.step()

        return loss.item(), tr_acc
