### This is a pytorch implementation of AGEM based on https://github.com/facebookresearch/agem. 

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import ipdb
import model.meta.learner as Learner
import model.meta.modelfactory as mf
import numpy as np
import random

from model.resnet import ResNet18
from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall


@dataclass
class AgemConfig:
    arch: str = "linear"
    n_layers: int = 2
    n_hiddens: int = 100
    memory_strength: float = 0.0
    dataset: str = "tinyimagenet"
    glances: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    memories: int = 5120
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 2.0
    input_channels: int = 1

    @staticmethod
    def from_args(args: object) -> "AgemConfig":
        cfg = AgemConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg
# Auxiliary functions useful for AGEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1



def projectgrad(gradient, memories, margin=0.5, eps = 1e-3, oiter = 0):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """

    similarity = torch.nn.functional.cosine_similarity(gradient.t(), memories.t().mean(dim=0).unsqueeze(0))

    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

    # merge memories
    t = memories_np.shape[0]

    memories_np2 = memories_np.mean(axis=0).reshape(1, memories_np.shape[1])

    ref_mag = np.dot(memories_np2, memories_np2.transpose())
    dotp = np.dot(gradient_np.reshape(1, -1), memories_np2.transpose())

    if(oiter%100==0):
        print('similarity : ', similarity.item())
        print('dotp:', dotp)

    if(dotp[0,0]<0):
        proj = gradient_np.reshape(1, -1) - ((dotp/ ref_mag) * memories_np2)
        gradient.copy_(torch.Tensor(proj).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = AgemConfig.from_args(args)

        nl, nh = self.cfg.n_layers, self.cfg.n_hiddens
        self.margin = self.cfg.memory_strength
        self.is_cifar = (
            (self.cfg.dataset == 'cifar100') or (self.cfg.dataset == 'tinyimagenet')
        )
        
        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch == 'resnet18':
            self.net = ResNet18(n_outputs, args)
            self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)
        elif self.cfg.arch == 'resnet1d':
            self.net = ResNet1D(n_outputs, args)
            self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)
        else:
            config = mf.ModelFactory.get_model(
                model_type = self.cfg.arch, 
                sizes = [n_inputs] + [nh] * nl + [n_outputs],
                dataset = self.cfg.dataset, args=args)
            self.net = Learner.Learner(config, args)

        self.ce = nn.CrossEntropyLoss()
        self.bce = torch.nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.glances = self.cfg.glances

        self.opt = optim.SGD(self.parameters(), self.cfg.lr)

        self.n_memories = self.cfg.n_memories
        self.gpu = self.cfg.cuda

        self.age = 0
        self.M = []
        self.memories = self.cfg.memories
        self.grad_align = []
        self.grad_task_align = {}
        self.current_task = 0

        # --- Episodic memory allocation ---
        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            # (task, mem, C=2, L)
            self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, 2, self.seq_len)
        else:
            # (task, mem, F)
            self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)

        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = int(n_outputs / n_tasks)
            # self.nc_per_task = n_outputs
        
        if self.gpu:
            self.cuda()

        self.iter = 0

    def _ensure_iq_shape(self, x):
        """
        Ensure x is (B, 2, L) for IQ mode.
        Accepts (B, 2, L) or (B, 2L).
        """
        if x.dim() == 3:
            # (B, 2, L) already
            return x
        elif x.dim() == 2:
            # (B, 2L) -> (B, 2, L)
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            L = F // 2
            return x.view(B, 2, L)
        else:
            raise ValueError(f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L).")
    
    def forward(self, x, t):
        if self.cfg.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x) # (B, 2, L)

        output = self.net.forward(x)

        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):

        self.iter +=1
        
        # --- shape handling ---
        if self.is_iq:
            # keep (B, 2, L)
            x = self._ensure_iq_shape(x)
        else:
            # legacy: flatten non-IQ inputs
            x = x.view(x.size(0), -1)

        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            self.grad_align.append([])
            
        tr_acc = []
        for pass_itr in range(self.glances):
# copy x into memory with matching shape
                
            if(pass_itr==0):
                # Update ring buffer storing examples from current task
                bsz = y.data.size(0)
                endcnt = min(self.mem_cnt + bsz, self.n_memories)
                effbsz = endcnt - self.mem_cnt
                # self.memory_data[t, self.mem_cnt: endcnt].copy_(
                #     x.data[: effbsz])
                # if bsz == 1:
                #     self.memory_labs[t, self.mem_cnt] = y.data[0]
                # else:
                #     self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                #         y.data[: effbsz])
                # self.mem_cnt += effbsz
                # if self.mem_cnt == self.n_memories:
                #     self.mem_cnt = 0

                if effbsz > 0:
                    if self.is_iq:
                        # shapes: mem slice (effbsz, 2, L) <- x[:effbsz]
                        self.memory_data[t, self.mem_cnt:endcnt].copy_(x.data[:effbsz])
                    else:
                        self.memory_data[t, self.mem_cnt:endcnt].copy_(x.data[:effbsz])

                    if bsz == 1:
                        self.memory_labs[t, self.mem_cnt] = y.data[0]
                    else:
                        self.memory_labs[t, self.mem_cnt:endcnt].copy_(y.data[:effbsz])

                    self.mem_cnt += effbsz
                    if self.mem_cnt == self.n_memories:
                        self.mem_cnt = 0

            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1:
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]

                    offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                       self.is_cifar)
                    logits = self.forward(Variable(self.memory_data[past_task]), past_task)[:, offset1: offset2]
                    ptloss = self.ce(
                        logits,
                        Variable(self.memory_labs[past_task] - offset1))
                    ptloss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)

            # now compute the grad on the current minibatch
            self.zero_grad()
            offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            logits = self.forward(x, t)[:, offset1: offset2]
            pb = torch.argmax(logits, dim=1)
            targets = y - offset1
            tr_acc.append(macro_recall(pb, targets))
            loss = self.ce(logits, targets)
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            # check if gradient violates constraints                                                           
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])

                projectgrad(self.grads[:, t].unsqueeze(1),                                           
                              self.grads.index_select(1, indx), self.margin, oiter = self.iter)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

            self.opt.step()
        
        xi = x.data.cpu().numpy()
        yi = y.data.cpu().numpy()
        for i in range(0,x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i],yi[i],t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi[i],yi[i],t]

        avg_tr_acc = sum(tr_acc)/len(tr_acc) if tr_acc else 0.0
        return loss.item(), avg_tr_acc
