# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.autograd import Variable
# from .common import ContextMLP, ContextNet18
# from .resnet import ResNet18 as ResNet18Full
from model.ctn_base import ContextNet18
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class CtnConfig:
    memory_strength: float = 0.5
    temperature: float = 5.0
    task_emb: int = 64
    lr: float = 0.01
    beta: float = 0.05
    n_memories: int = 50
    validation: float = 0.0
    cuda: bool = True
    batch_size: int = 1
    samples_per_task: int = -1
    replay_batch_size: int = 20
    inner_steps: int = 2
    n_meta: int = 2

    @staticmethod
    def from_args(args: object) -> "CtnConfig":
        cfg = CtnConfig()
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
        self.cfg = CtnConfig.from_args(args)
        self.reg = self.cfg.memory_strength
        self.temp = self.cfg.temperature
        # setup network
        if  self.cfg.arch == 'resnet1d':
            # self.net = ResNet1D(n_outputs, args)
            self.net = ContextNet18(n_outputs, n_tasks=n_tasks, task_emb=self.cfg.task_emb)
        # self.net.define_task_lr_params(alpha_init=args.alpha_init)
        else:
            raise NotImplementedError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")

        self.is_task_incremental = True 
        self.inner_lr = self.cfg.lr
        self.outer_lr = self.cfg.beta
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr, momentum=0.9)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        if self.is_task_incremental:
            self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = self.cfg.n_memories
        self.mem_cnt = 0       
        
        # set up the semantic memory
        self.n_val = int(self.n_memories * self.cfg.validation)
        self.n_memories -= self.n_val
        self.full_val = True # avoid OOM when using too large memory

        # if 'cub' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val, 3, 224, 224)
        # elif 'mini' in args.data_file or 'core' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 84, 84)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val , 3, 84, 84)
        #     if self.n_memories > 75:
        #         self.full_val = False
        # else:
        self.memx = torch.FloatTensor(n_tasks, self.n_memories, 2, n_inputs//2)
        self.valx = torch.FloatTensor(n_tasks, self.n_val, 2, n_inputs//2)

        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.valy = torch.LongTensor(n_tasks, self.n_val)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}
        if self.cfg.cuda:
            self.valx = self.valx.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.memy = self.memy.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)
            #self.valy.data.fill_(0)

        self.mem_cnt = 0
        self.val_cnt = 0
        self.bsz = self.cfg.batch_size
        
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        # Use batchmean to align with KL definition and silence PyTorch deprecation warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.samples_per_task = self.cfg.samples_per_task
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps
        self.n_meta = self.cfg.n_meta
        self.count = 0
        self.val_count = 0
        self.counter = 0
    def on_epoch_end(self):  
        self.counter += 1
        pass

    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def forward(self, x, t, return_feat= False):
        output = self.net(x, t)
        
        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t, valid = False):
        if valid:
            mem_x = self.valx[:t+1,:]
            mem_y = self.valy[:t+1,:]
            mem_feat = self.mem_feat[:t,:]
            if self.full_val:
                idx = np.arange(t*mem_y.size(1))
            else:
                sz = min(t*mem_y.size(1), 64)
                idx = np.random.choice(t* mem_y.size(1) ,sz, False)
            t_idx = torch.from_numpy(idx // self.n_val)
            s_idx = torch.from_numpy( idx % self.n_val)
            offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
            xx = mem_x[t_idx, s_idx]
            yy = mem_y[t_idx, s_idx] - offsets[:,0]
            mask = torch.zeros(xx.size(0), self.nc_per_task)
            for j in range(mask.size(0)):
                cls_size = offsets[j][1] - offsets[j][0]
                mask[j, :cls_size] = torch.arange(offsets[j][0], offsets[j][1])
            sizes = (offsets[:, 1] - offsets[:, 0]).long()
            return xx,yy, 0 , mask.long().cuda(), t_idx.tolist(), sizes
        else:
            mem_x = self.memx[:t,:]
            mem_y = self.memy[:t,:]
            mem_feat = self.mem_feat[:t,:]
            sz = int(min(self.n_memories, self.sz))
            idx = np.random.choice(int(t * self.n_memories), sz, False)
            t_idx = torch.from_numpy(idx // self.n_memories)
            s_idx = torch.from_numpy( idx % self.n_memories)
            offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
            xx = mem_x[t_idx, s_idx]
            yy = mem_y[t_idx, s_idx] - offsets[:,0]
            feat = mem_feat[t_idx, s_idx]
            mask = torch.zeros(xx.size(0), self.nc_per_task)
            for j in range(mask.size(0)):
                cls_size = offsets[j][1] - offsets[j][0]
                mask[j, :cls_size] = torch.arange(offsets[j][0], offsets[j][1])
            sizes = (offsets[:, 1] - offsets[:, 0]).long()
            return xx,yy, feat , mask.long().cuda(), t_idx.tolist(), sizes
        
    def observe(self, x, y, t):

        # if task has changed, run model on val set of previous task to get soft targets 
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone() # store soft targets
            self.current_task = t
            self.mem_cnt = 0
            self.val_cnt = 0
            self.val_count = 0
            self.memy[t] = 0
            self.count=0

        # maintain validation set
        valx = x[0]
        valy = y[0]
        x = x[1:]
        y = y[1:]
        if self.val_cnt == 0 and self.val_count == 0:
            self.valx[t,:].copy_(valx)
            self.valy[t,:].copy_(valy)
        else:    
            x = torch.cat([x, self.valx[t,self.val_cnt-1].unsqueeze_(0)])
            y = torch.cat([y, self.valy[t,self.val_cnt-1].unsqueeze_(0)])
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
        meta_grad_init = [0 for _ in range(len(self.net.state_dict()))]
        #for _ in range(self.inner_steps):
        tr_acc = []
        for _ in range(self.n_meta):
            meta_grad = deepcopy(meta_grad_init)
            loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.).cuda()
            loss3 = torch.tensor(0.).cuda()
 
            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x,t)
            logits = pred[:, offset1:offset2]
            targets = y - offset1
            preds = torch.argmax(logits, dim=1)
            tr_acc.append(macro_recall(preds, targets))

            loss1 = self.bce(logits, targets)
            #tt = t + 1
            for i in range(self.inner_steps):
                if t > 0:
                    xx, yy, feat, mask, list_t, class_sizes = self.memory_sampling(t)
                    pred_ = self.net(xx, list_t)
                    pred = torch.gather(pred_, 1, mask)
                    for row, size in enumerate(class_sizes):
                        if size < pred.size(1):
                            pred[row, size:] = -1e9
                    loss2 = self.bce(pred, yy)
                    loss3 = self.reg * self.kl(F.log_softmax(pred / self.temp, dim = 1), feat)
                    loss = loss1 + loss2 + loss3
                else:
                    loss = loss1
             
                grads = torch.autograd.grad(loss, self.net.base_param(), create_graph=True)
                
                # SGD update only the BASE NETWORK
                for param, grad in zip(self.net.base_param(), grads):
                    new_param = param.data.clone()
                    new_param = new_param - self.inner_lr * grad
                    param.data.copy_(new_param)

            xval, yval, feat, mask, list_t, class_sizes_val = self.memory_sampling(t+1, valid = True)
            pred_ = self.net(xval, list_t)
            pred = torch.gather(pred_, 1, mask)
            for row, size in enumerate(class_sizes_val):
                if size < pred.size(1):
                    pred[row, size:] = -1e9
            outer_loss = self.bce(pred, yval)
            outer_grad = torch.autograd.grad(outer_loss, self.net.context_param())
                
            for g in range(len(outer_grad)):
                meta_grad[g] += outer_grad[g].detach()

            self.opt.zero_grad()
            for c, param in enumerate(self.net.context_param()):
                param.grad = meta_grad[c] / float(self.n_meta)
                param.grad.data.clamp_(-1,1)
            self.opt.step()
            #SGD update the CONTROLLER 
            self.zero_grad() 
               
        avg_tr_acc = sum(tr_acc) / len(tr_acc) if tr_acc else 0.0
        return loss.item(), avg_tr_acc
