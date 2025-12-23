# An implementation of MER Algorithm 1 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file found in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import ipdb
import warnings
warnings.filterwarnings("ignore")
from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class MerAlgConfig:
    arch: str = "resnet1d"
    n_layers: int = 2
    n_hiddens: int = 100
    dataset: str = "tinyimagenet"
    alpha_init: float = 1e-3
    lr: float = 1e-3
    replay_batch_size: int = 20
    memories: int = 5120
    batches_per_example: int = 1
    beta: float = 1.0
    gamma: float = 0.0
    cuda: bool = True
    grad_clip_norm: Optional[float] = 2.0
    input_channels: int = 1

    @staticmethod
    def from_args(args: object) -> "MerAlgConfig":
        cfg = MerAlgConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = MerAlgConfig.from_args(args)
        self.is_cifar = (self.cfg.dataset == 'cifar100' or self.cfg.dataset == 'tinyimagenet')

        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.netforward = self.net.forward

        self.bce = torch.nn.CrossEntropyLoss()

        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        # if self.is_cifar:
        #     self.nc_per_task = n_outputs / n_tasks
        # else:
        #     self.nc_per_task = n_outputs

        self.opt = optim.SGD(self.parameters(), self.cfg.lr)
        self.batchSize = int(self.cfg.replay_batch_size)

        self.memories = self.cfg.memories
        self.steps = int(self.cfg.batches_per_example)
        self.beta = self.cfg.beta
        self.gamma = self.cfg.gamma

        # allocate buffer
        self.M = []
        self.age = 0

        # handle gpus if specified
        self.cuda = self.cfg.cuda
        if self.cuda:
            self.net = self.net.cuda()


    def forward(self, x, t):
        output = self.netforward(x)
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def _clone_model_state(self):
        # Use the underlying model to keep state_dict keys aligned with _apply_meta_update
        return {name: tensor.detach().clone() for name, tensor in self.net.model.state_dict().items()}

    def _apply_meta_update(self, base_state, target_state, mix):
        own_params = dict(self.net.model.named_parameters())
        own_params.update(dict(self.net.model.named_buffers()))
        with torch.no_grad():
            for name, tensor in own_params.items():
                tensor.copy_(base_state[name] + (target_state[name] - base_state[name]) * mix)

    def getBatch(self,x,y,t):
        samples = []
        if x is not None:
            samples.append((x, y, t))
        if len(self.M) > 0:
            osize = min(self.batchSize,len(self.M))
            indices = random.sample(range(len(self.M)), osize)
            for idx in indices:
                samples.append(self.M[idx])

        bxs = []
        bys = []
        bts = []
        for sx, sy, st in samples:
            bx = sx.unsqueeze(0).float()
            by = sy.view(1).long()
            if self.cuda:
                bx = bx.cuda(non_blocking=True)
                by = by.cuda(non_blocking=True)
            bxs.append(bx)
            bys.append(by)
            bts.append(st)

        return bxs,bys,bts
               

    def observe(self, x, y, t):

        # step through elements of x
        batch_preds = []
        batch_targets = []

        task_id = int(t) if isinstance(t, int) else int(t.item())
        for i in range(0,x.size()[0]):

            self.age += 1
            xi = x[i].detach().cpu()
            yi = y[i].detach().cpu()
            self.net.zero_grad()

            before = self._clone_model_state()
            for step in range(0,self.steps):
                weights_before = self._clone_model_state()
                ##Check for nan
                if weights_before != weights_before:
                    ipdb.set_trace()
                # Draw batch from buffer:
                bxs, bys, bts = self.getBatch(xi,yi,task_id)          
                loss = 0.0
                total_loss = 0.0
                for idx in range(len(bxs)):

                    self.net.zero_grad()
                    bx = bxs[idx] 
                    by = bys[idx] 
                    bt = bts[idx]

                    if self.is_cifar:
                        offset1, offset2 = self.compute_offsets(bt)
                        prediction = (self.netforward(bx)[:, offset1:offset2])
                        loss = self.bce(prediction,
                                        by - offset1)
                        preds = torch.argmax(prediction, dim=1)
                        target = by - offset1
                    else:
                        prediction = self.forward(bx,0)
                        loss = self.bce(prediction, by)
                        preds = torch.argmax(prediction, dim=1)
                        target = by
                    if torch.isnan(loss):
                        ipdb.set_trace()

                    loss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
                    self.opt.step()
                    batch_preds.append(preds.detach().cpu())
                    batch_targets.append(target.detach().cpu())
                    total_loss += loss.item()
                weights_after = self._clone_model_state()
                if weights_after != weights_after:
                    ipdb.set_trace()

                # Within batch Reptile meta-update:
                self._apply_meta_update(weights_before, weights_after, self.beta)

            after = self._clone_model_state()

            # Across batch Reptile meta-update:
            self._apply_meta_update(before, after, self.gamma)

            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append((xi.clone(), yi.clone(), task_id))

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = (xi.clone(), yi.clone(), task_id)

        if batch_preds:
            stacked_preds = torch.stack(batch_preds).view(-1)
            stacked_targets = torch.stack(batch_targets).view(-1)
            avg_tr_acc = macro_recall(stacked_preds, stacked_targets)
        else:
            avg_tr_acc = 0.0
        return total_loss/self.steps, avg_tr_acc
