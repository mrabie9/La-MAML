# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np
import random

import model.meta.learner as Learner
import model.meta.modelfactory as mf
import sys
from model.resnet import ResNet18
from model.resnet1d import ResNet1D
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataloaders')))
from iq_data_loader import ensure_iq_two_channel

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("once")

class Net(torch.nn.Module):
    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.args = args
        self.nt = n_tasks
        self.reg = args.memory_strength
        self.n_memories = args.n_memories
        self.num_exemplars = 0
        self.n_feat = n_outputs
        self.n_classes = n_outputs
        self.samples_per_task = args.samples_per_task * (1.0 - args.validation)
        assert self.samples_per_task > 0, 'Samples per task is <= 0'
        self.examples_seen = 0

        self.glances = args.glances
        # setup network

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

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss()  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        # memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}

        self.gpu = args.cuda
        self.nc_per_task = int(n_outputs / n_tasks)
        self.n_outputs = n_outputs

    def netforward(self, x):
        if self.args.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.args.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif 'iq' in self.args.dataset.lower():
            # print(x.shape)
            x = ensure_iq_two_channel(x.detach().cpu().numpy())
            x = torch.from_numpy(x).float().cuda()
            # print(x.shape)

        return self.net.forward(x)

    def compute_offsets(self, task):
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def forward(self, x, t):
        # nearest neighbor
        nd = self.n_feat
        ns = x.size(0)
        if t * self.nc_per_task not in self.mem_class_x.keys():
            # no exemplar in memory yet, output uniform distr. over classes in
            # task t above, we check presence of first class for this task, we
            # should check them all
            out = torch.Tensor(ns, self.n_classes).fill_(-10e10)
            out[:, int(t * self.nc_per_task): int((t + 1) * self.nc_per_task)].fill_(
                1.0 / self.nc_per_task)
            if self.gpu:
                out = out.cuda()
            return out
        means = torch.ones(self.nc_per_task, nd) * float('inf')
        if self.gpu:
            means = means.cuda()
        offset1, offset2 = self.compute_offsets(t)
        for cc in range(offset1, offset2):
            means[cc -
                  offset1] =self.netforward(self.mem_class_x[cc]).data.mean(0)
        classpred = torch.LongTensor(ns)
        preds = self.netforward(x).data.clone()
        for ss in range(ns):
            dist = (means - preds[ss].expand(self.nc_per_task, nd)).norm(2, 1)
            _, ii = dist.min(0)
            ii = ii.squeeze()
            classpred[ss] = ii.item() + offset1

        out = torch.zeros(ns, self.n_classes)
        if self.gpu:
            out = out.cuda()
        for ss in range(ns):
            out[ss, classpred[ss]] = 1
        return out  # return 1-of-C code, ns x nc

    def forward_training(self, x, t):
        output = self.netforward(x)
        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)

        # zero out all the logits outside the task's range
        # since the output vector from the model is of dimension (num_tasks * num_classes_per_task)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):

        x = x.view(x.size(0), -1)
        self.net.train()

        for pass_itr in range(self.glances):

            # only make changes like pushing to buffer once per batch and not for every glance
            if(pass_itr==0):
                self.examples_seen += x.size(0)

                # if not last batch of task, store samples in memx/memy
                if self.examples_seen < self.samples_per_task:
                    if self.memx is None:
                        self.memx = x.data.clone()
                        self.memy = y.data.clone()
                    else:
                        self.memx = torch.cat((self.memx, x.data.clone()))
                        self.memy = torch.cat((self.memy, y.data.clone()))

            self.net.zero_grad()
            offset1, offset2 = self.compute_offsets(t)
            loss = self.bce((self.netforward(x)[:, offset1: offset2]),
                            y - offset1)
            # num_exemplars remains 0 unless final epoch is reached
            if self.num_exemplars > 0:
                # distillation
                for tt in range(t):
                    # first generate a minibatch with one example per class from
                    # previous tasks
                    inp_dist = torch.zeros(self.nc_per_task, x.size(1))
                    target_dist = torch.zeros(self.nc_per_task, self.n_feat)
                    offset1, offset2 = self.compute_offsets(tt)
                    if self.gpu:
                        inp_dist = inp_dist.cuda()
                        target_dist = target_dist.cuda()
                    for cc in range(self.nc_per_task):
                        indx = random.randint(0, len(self.mem_class_x[cc + offset1]) - 1)
                        inp_dist[cc] = self.mem_class_x[cc + offset1][indx].clone()
                        target_dist[cc] = self.mem_class_y[cc +
                                                           offset1][indx].clone()
                    # Add distillation loss
                    loss += self.reg * self.kl(
                        self.lsm(self.netforward(inp_dist)
                                 [:, offset1: offset2]),
                        self.sm(target_dist[:, offset1: offset2])) * self.nc_per_task
            # bprop and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)

            self.opt.step()

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!
        target = int(self.args.n_epochs * self.samples_per_task)
        if self.examples_seen >= target:  # not ==
            self.examples_seen = 0
            # self._rebuild_exemplars_for_task(t, x.device)

        # if self.examples_seen == self.args.n_epochs * self.samples_per_task:
        #     self.examples_seen = 0
            # get labels from previous task; we assume labels are consecutive
            offset1, offset2 = self.compute_offsets(t)
            if self.gpu:
                all_labs = torch.LongTensor(np.unique(self.memy.cpu().numpy()))
            else:
                all_labs = torch.LongTensor(np.unique(self.memy.numpy()))
            
            if self.args.loader == 'class_incremental_loader':
                num_classes = all_labs.size(0)
            else:
                num_classes = all_labs[offset1:offset2].size(0)

            assert(num_classes == self.nc_per_task)
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(self.n_memories /
                                     (num_classes + len(self.mem_class_x.keys())))
            for ll in range(num_classes):
                label = all_labs[ll]#.cuda() # current label
                indxs = (self.memy == label).nonzero().squeeze() # indices of current label
                cdata = self.memx.index_select(0, indxs) # grab training data for current label
                # Construct exemplar set for last task
                mean_feature = self.netforward(cdata)[
                    :, offset1: offset2].data.clone().mean(0) # mean of task-sliced logits for current label
                nd = self.nc_per_task # num classes per task
                exemplars = torch.zeros(self.num_exemplars, x.size(1))
                if self.gpu:
                    exemplars = exemplars.cuda()
                ntr = cdata.size(0) # num data points for current label
                # used to keep track of which examples we have already used
                taken = torch.zeros(ntr)
                model_output = self.netforward(cdata)[
                    :, offset1: offset2].data.clone() # clone model output for current label
                for ee in range(self.num_exemplars): # herding loop
                    prev = torch.zeros(1, nd)
                    if self.gpu:
                        prev = prev.cuda()
                    if ee > 0:
                        prev = self.netforward(exemplars[:ee])[
                            :, offset1: offset2].data.clone().sum(0)
                    cost = (mean_feature.expand(ntr, nd) - (model_output
                                                            + prev.expand(ntr, nd)) / (ee + 1)).norm(2, 1).squeeze()
                    _, indx = cost.sort(0) # sort by ascending cost
                    winner = 0
                    while winner < indx.size(0) and taken[indx[winner]] == 1:
                        winner += 1
                    if winner < indx.size(0):
                        taken[indx[winner]] = 1
                        exemplars[ee] = cdata[indx[winner]].clone()
                    else:
                        exemplars = exemplars[:indx.size(0), :].clone()
                        self.num_exemplars = indx.size(0)
                        break
                # update memory with exemplars
                self.mem_class_x[label.item()] = exemplars.clone()

            # recompute outputs for distillation purposes
            for cc in self.mem_class_x.keys():
                self.mem_class_x[cc] = self.mem_class_x[cc][:self.num_exemplars]
                self.mem_class_y[cc] = self.netforward(
                    self.mem_class_x[cc]).data.clone()
            self.memx = None
            self.memy = None
            print(len(self.mem_class_x[0]))

        return loss.item()