# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from model.resnet1d import ResNet1D
# from .resnet import ResNet18 as ResNet18Full
import pdb

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        self.reg = args.memory_strength

        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=args.alpha_init)

        # setup optimizer
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        if self.is_task_incremental:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

    def compute_offsets(self, task):
        if self.is_task_incremental:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    def on_epoch_end(self):
        pass

    def observe(self, x, y, t):
        self.net.train()

        # next task?
        if t != self.current_task:
            self.net.zero_grad()

            if self.is_task_incremental:
                offset1, offset2 = self.compute_offsets(self.current_task)
                self.bce((self.net(self.memx)[:, offset1: offset2]),
                         self.memy - offset1).backward()
            else:
                self.bce(self(self.memx,
                              self.current_task),
                         self.memy).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.is_task_incremental:
            offset1, offset2 = self.compute_offsets(t)
            loss = self.bce((self.net(x)[:, offset1: offset2]),
                            y - offset1)
        else:
            loss = self.bce(self(x, t), y)
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()
        return loss.item()
