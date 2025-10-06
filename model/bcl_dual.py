# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from model.resnet1d import ResNet1D
from .resnet import ResNet18 as ResNet18Full
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.n_tasks = n_tasks
        self.reg = args.bcl_memory_strength
        self.temp = args.bcl_temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=args.alpha_init)

        # setup optimizer
        self.inner_lr = args.lr
        self.beta= args.beta
        #self.outer_opt = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr)
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
        
        self.n_val = int(self.n_memories * 0.2)
        self.n_memories -= self.n_val
        
        self.memx = torch.FloatTensor(n_tasks, self.n_memories, 2, n_inputs//2)
        self.valx = torch.FloatTensor(n_tasks, self.n_val, 2, n_inputs//2)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task).fill_(0)
        self.valy = torch.LongTensor(n_tasks, self.n_val)
        self.mem = {}
        if args.cuda:
            self.memy = self.memy.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda()
            self.valx = self.valx.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)

        self.mem_cnt = 0
        self.val_cnt = 0
        self.bsz = args.batch_size
        self.valid_id = []
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.bcl_inner_steps
        self.n_meta = args.bcl_n_meta
        self.count = 0
        self.val_count = 0
        self.adapt_ = False #args.adapt
        self.adapt_lr = args.bcl_adapt_lr
        self.models={}

    def on_epoch_end(self):  
        pass
        
    def adapt(self):
        print('Adapting')
        for t in range(self.n_tasks):
            model = deepcopy(self.net)
            if t > self.current_task:
                self.models[t] = model
                continue
            xx = self.memx[t]
            yy = self.memy[t]
            opt = torch.optim.SGD(model.parameters(), self.adapt_lr)
            train = torch.utils.data.TensorDataset(xx, yy)
            loader = DataLoader(train, batch_size = self.bsz, shuffle = True, num_workers =0)
            for _ in range(self.inner_steps):
                model.zero_grad()
                pred = model.forward(xx)
                loss = self.bce(pred, yy)
                loss.backward()
                opt.step()
            self.models[t] = model


    def compute_offsets(self, task):
        if self.is_task_incremental:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        if self.adapt_ and not self.net.training:
            output = self.models[t](x)
        else:
            output = self.net(x)
        
        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t, valid = False):
        '''  
        if t == self.current_task:
            val_idx = self.val_count
            mem_idx = self.count
        else:
            val_idx = self.n_val + 1
            mem_idx = self.count + 1
        t + = 1
        '''
        if valid:
            mem_x = self.valx[:t,:]
            mem_y = self.valy[:t,:]
            mem_feat = self.mem_feat[:t,:]
            sz = min(t*self.n_val, self.sz)
            idx = np.random.choice(t* self.n_val,sz, False)
            self.valid_id = idx.tolist()
            t_idx = torch.from_numpy(idx // self.n_val)
            s_idx = torch.from_numpy( idx % self.n_val)
        else:
            mem_x = self.memx[:t,:]
            mem_y = self.memy[:t,:]
            mem_feat = self.mem_feat[:t,:]
            sz = min(self.n_memories, self.sz)
            idx = np.random.choice(t* self.n_memories,sz, False)
            idx = [x for x in idx if x not in self.valid_id]
            idx = np.array(idx)
            t_idx = torch.from_numpy(idx // self.n_memories)
            s_idx = torch.from_numpy( idx % self.n_memories)

        offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:,0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        mask = mask.long().cuda()
        return xx,yy, feat , mask, t_idx.tolist()
    def observe(self, x, y, t):
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t
            self.mem_cnt = 0
            self.val_cnt = 0
            self.val_count = 0
            self.memy[t] = 0
            self.count=0
        
        # evalidation set
        valx, valy = x[0], y[0]
        x, y = x[1:], y[1:]
        if self.val_cnt == 0 and self.val_count == 0:
            self.valx[t,:].copy_(valx)
            self.valy[t,:].copy_(valy)
        else:
            x = torch.cat([x, self.valx[t,self.val_cnt].unsqueeze_(0)])
            y = torch.cat([y, self.valy[t,self.val_cnt].unsqueeze_(0)])
            self.valx[t, self.val_cnt].copy_(valx)
            self.valy[t, self.val_cnt].copy_(valy)
        self.val_cnt += 1
        self.val_count += 1
        if self.val_count == self.n_val:
            self.val_count -= 1
        if self.val_cnt == self.n_val:
            self.val_cnt = 0

        # memory set
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt        
        if self.count == 0:
            self.memx[t,:].copy_(x.data[0])
            self.memy[t,:].copy_(y.data[0])
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        self.count += effbsz
        if self.count >= self.n_memories:
            self.count -= effbsz
        if self.mem_cnt >= self.n_memories:
            self.mem_cnt = 0

        self.zero_grad()   
        tt = t + 1
        for _ in range(self.n_meta):
            weights_before = deepcopy(self.net.state_dict())
            offset1, offset2 = self.compute_offsets(t)
            for i in range(self.inner_steps):
                pred = self.forward(x,t)
                loss1 = self.bce(pred[:, offset1:offset2], y - offset1)
                if t > 0:
                    xx, yy, feat, mask, list_t = self.memory_sampling(t)
                    pred_ = self.net(xx)
                    pred = torch.gather(pred_, 1, mask)
                    loss2 = self.bce(pred, yy)
                    loss3 = self.reg * self.kl(F.log_softmax(pred / self.temp, dim = 1), feat)
                    loss = loss1 + loss2 + loss3
                else:
                    loss = loss1
                loss.backward()
                self.inner_opt.step()
            xval, yval, _, mask_val, list_t = self.memory_sampling(tt, valid = True)  
            pred_ = self.net(xval)
            pred = torch.gather(pred_, 1, mask_val)
            outer_loss = self.bce(pred, yval)
            outer_loss.backward()                    
            self.inner_opt.step()
            self.zero_grad()
            weights_after = self.net.state_dict()
            new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before.keys()}
            self.net.load_state_dict(new_params)
        return outer_loss.item()
