import random
from random import shuffle
import numpy as np
import ipdb
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from model.resnet1d import ResNet1D
from scipy.stats import pearsonr
import datetime
from utils import misc_utils


@dataclass
class LamamlBaseConfig:
    alpha_init: float = 1e-3
    opt_wt: float = 1e-1
    opt_lr: float = 1e-1
    glances: int = 1
    memories: int = 5120
    replay_batch_size: int = 20
    cuda: bool = True
    use_old_task_memory: bool = False
    learn_lr: bool = False
    second_order: bool = False
    sync_update: bool = False
    meta_batches: int = 3
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    grad_clip_norm: Optional[float] = 2.0
    n_layers: int = 2
    n_hiddens: int = 100
    input_channels: int = 1

    @staticmethod
    def from_args(args: object) -> "LamamlBaseConfig":
        cfg = LamamlBaseConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class BaseNet(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(BaseNet, self).__init__()

        self.args = args
        self.cfg = LamamlBaseConfig.from_args(args)
        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.opt_wt = torch.optim.SGD(list(self.net.parameters()), lr=self.cfg.opt_wt, momentum=0.9)
        self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=self.cfg.opt_lr, momentum=0.9)

        self.epoch = 0
        # allocate buffer
        self.M = []        
        self.M_new = []
        self.age = 0

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()
        self.is_cifar = ((self.cfg.dataset == 'cifar100') or (self.cfg.dataset == 'tinyimagenet'))
        self.glances = self.cfg.glances
        self.pass_itr = 0
        self.real_epoch = 0

        self.current_task = 0
        self.memories = self.cfg.memories
        self.batchSize = int(self.cfg.replay_batch_size)

        self.use_cuda = self.cfg.cuda
        if self.use_cuda:
            self.net = self.net.cuda()

        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()              
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]


    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch of old + new data, 
        where old data is sampled from the memory buffer
        """

        if(x is not None):
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int)*t        
        else:
            mxi = np.empty( shape=(0, 0) )
            myi = np.empty( shape=(0, 0) )
            mti = np.empty( shape=(0, 0) )    

        bxs = []
        bys = []
        bts = []

        if self.cfg.use_old_task_memory and t>0:
            MEM = self.M
        else:
            MEM = self.M_new

        batch_size = self.batchSize if batch_size is None else batch_size

        # Sample from memory buffer if not empty
        if len(MEM) > 0:
            order = [i for i in range(0,len(MEM))]
            osize = min(batch_size,len(MEM))
            for j in range(0,osize):

                # randomly sample from self.M_new memory buffer
                shuffle(order)
                k = order[j]
                x,y,t = MEM[k]

                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # add new data points - ratio of old to new becomes up to 1:1
        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float() 
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs,bys,bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def zero_grads(self):
        if self.cfg.learn_lr:
            self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()
        
