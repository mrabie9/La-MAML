# TODO: update mmemory buffer to store IQ values together

### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory. 
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import model.meta.learner as Learner
import model.meta.modelfactory as mf
import numpy as np
import quadprog

from model.resnet import ResNet18
from model.resnet1d import ResNet1D

# Auxiliary functions useful for GEM's inner optimization.

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



def project2cone2(gradient, memories, margin=0.5, eps = 1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())  + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.args = args
        self.margin = args.memory_strength
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))

        # --- IQ mode toggle ---
        self.input_channels = getattr(args, "input_channels", 1)
        self.is_iq = (getattr(args, "dataset", "") == "iq") or (self.input_channels == 2)

        nl, nh = args.n_layers, args.n_hiddens

        if args.arch == 'resnet18':
            self.net = ResNet18(n_outputs, args)
            self.net.define_task_lr_params(alpha_init=args.alpha_init)
        elif args.arch == 'resnet1d':
            self.net = ResNet1D(n_outputs, args)
            self.net.define_task_lr_params(alpha_init=args.alpha_init)
        else:
            config = mf.ModelFactory.get_model(
                model_type=args.arch,
                sizes=[n_inputs] + [nh] * nl + [n_outputs],
                dataset=args.dataset, args=args
            )
            self.net = Learner.Learner(config, args=args)

        self.netforward = self.net.forward
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.glances = args.glances

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

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
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # --- GEM gradient buffers ---
        self.grad_dims = [p.data.numel() for p in self.parameters()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # --- counters / bookkeeping ---
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = int(n_outputs / n_tasks)

        if args.cuda:
            self.cuda()

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
        if self.args.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.args.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x)  # (B, 2, L)

        output = self.netforward(x)

        if self.is_cifar:
            # class-masking within current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):
        """
        One optimization step on batch (x,y,t), with GEM constraints and glances.
        """
        # --- shape handling ---
        if self.is_iq:
            # keep (B, 2, L)
            x = self._ensure_iq_shape(x)
        else:
            # legacy: flatten non-IQ inputs
            x = x.view(x.size(0), -1)

        # track tasks
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        for pass_itr in range(self.glances):
            # push current batch once per batch (not each glance)
            if pass_itr == 0:
                bsz = y.data.size(0)
                endcnt = min(self.mem_cnt + bsz, self.n_memories)
                effbsz = endcnt - self.mem_cnt

                # copy x into memory with matching shape
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

            # gradients on past tasks (replay)
            if len(self.observed_tasks) > 1:
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    past_task = self.observed_tasks[tt]
                    offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)

                    # replay batch (shape already in memory)
                    mem_x = Variable(self.memory_data[past_task])          # (mem, F) or (mem, 2, L)
                    mem_y = Variable(self.memory_labs[past_task])          # (mem,)

                    ptloss = self.ce(
                        self.forward(mem_x, past_task)[:, offset1:offset2],
                        mem_y - offset1
                    )
                    ptloss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                    store_grad(self.parameters, self.grads, self.grad_dims, past_task)

            # current batch
            self.zero_grad()
            offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            loss = self.ce(self.forward(x, t)[:, offset1:offset2], y - offset1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            # GEM projection if needed
            if len(self.observed_tasks) > 1:
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu else torch.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)

            self.opt.step()
        return loss.item()