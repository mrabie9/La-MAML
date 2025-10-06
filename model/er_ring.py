# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from model.resnet1d import ResNet1D
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.reg = args.bcl_memory_strength
        self.temp = args.bcl_temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=args.alpha_init)
        # setup optimizer
        self.lr = args.lr
        #if self.is_task_incremental:
        #    self.opt = torch.optim.Adam(self.net.parameters(), lr='self.lr)
        #else:
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_task_incremental:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.bcl_n_memories
        self.mem_cnt = 0       
        
        self.memx = torch.FloatTensor(n_tasks, self.n_memories, 2, n_inputs//2)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}
        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()
        self.mem_cnt = 0
        self.n_memories = args.n_memories
        self.bsz = args.batch_size
        
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.bcl_inner_steps
    def on_epoch_end(self):  
        pass

    def compute_offsets(self, task):
        if self.is_task_incremental:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        output = self.net(x)
        
        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t):
        mem_x = self.memx[:t,:]
        mem_y = self.memy[:t,:]
        mem_feat = self.mem_feat[:t,:]
        sz = min(self.n_memories, self.sz)
        idx = np.random.choice(t* self.n_memories,sz, False)
        t_idx = torch.from_numpy(idx // self.n_memories)
        s_idx = torch.from_numpy( idx % self.n_memories)

        offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:,0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return xx,yy, feat , mask.long().cuda()
    def observe(self, x, y, t):
        #t = info[0]
        #idx = info[1]
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            #self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t
            
        for _ in range(self.inner_steps):
            self.net.zero_grad()
            loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.).cuda()
            loss3 = torch.tensor(0.).cuda()
         
            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x,t, True)
            loss1 = self.bce(pred[:, offset1:offset2], y - offset1)
            if t > 0:
                xx, yy, target, mask = self.memory_sampling(t)
                pred_ = self.net(xx)
                pred = torch.gather(pred_, 1, mask)
                loss2 += self.bce(pred, yy)
                
            loss = loss1 + loss2
            loss.backward()
            self.opt.step()

        return loss.item()
