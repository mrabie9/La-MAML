# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch

import numpy as np
import random

import sys
from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataloaders')))
from iq_data_loader import ensure_iq_two_channel
from utils.training_metrics import macro_recall
from utils import misc_utils

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("once")


@dataclass
class IcarlConfig:
    lr: float = 1e-3
    memory_strength: float = 0.0
    n_memories: int = 0
    glances: int = 1

    grad_clip_norm: Optional[float] = 100.0
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    n_epochs: int = 1
    input_channels: int = 2
    alpha_init: float = 1e-3
    samples_per_task: int = -1
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "IcarlConfig":
        cfg = IcarlConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(DetectionReplayMixin, torch.nn.Module):
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
        self.cfg = IcarlConfig.from_args(args)
        self.nt = n_tasks
        self.reg = self.cfg.memory_strength
        self.n_memories = self.cfg.n_memories
        self.num_exemplars = 0
        self.n_feat = n_outputs
        self.n_classes = n_outputs
        self.samples_per_task_resolver = getattr(args, "get_samples_per_task", None)
        self.samples_per_task = self.cfg.samples_per_task #* (1.0 - self.cfg.validation)
        if self.samples_per_task_resolver is None:
            assert self.samples_per_task > 0, 'Samples per task is <= 0'
        self.examples_seen = 0

        self.glances = self.cfg.glances
        # setup network

        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        # setup optimizer
        self.opt = torch.optim.SGD(self._ll_params(), lr=self.cfg.lr, momentum=0.9)
        self.det_opt = torch.optim.SGD(self.net.det_head.parameters(), lr=self.cfg.lr, momentum=0.9)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        # Use batchmean to follow KL definition and avoid PyTorch warning
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(self.cfg.det_memories, self.cfg.det_replay_batch)

        # memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}

        self.gpu = self.cfg.cuda
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.n_outputs = n_outputs

    def _ensure_iq_shape(self, x):
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            L = F // 2
            return x.view(B, 2, L)
        raise ValueError(f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L).")

    def _prepare_input(self, x):
        if self.cfg.dataset == 'tinyimagenet':
            return x.view(-1, 3, 64, 64)
        if self.cfg.dataset == 'cifar100':
            return x.view(-1, 3, 32, 32)
        if self.is_iq:
            return self._ensure_iq_shape(x)
        return x

    def _prepare_det_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._prepare_input(x)

    def netforward(self, x):
        if self.cfg.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif 'iq' in self.cfg.dataset.lower():
            # print(x.shape)
            x = ensure_iq_two_channel(x.detach().cpu().numpy())
            x = torch.from_numpy(x).float().cuda()
            # print(x.shape)

        return self.net.forward(x)

    def compute_offsets(self, task):
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def _get_samples_per_task(self, task):
        if self.samples_per_task_resolver is None:
            return self.samples_per_task
        return int(self.samples_per_task_resolver(task))

    def forward(self, x, t):
        # nearest neighbor
        nd = self.n_feat
        ns = x.size(0)
        task_classes = self.classes_per_task[t]
        offset1, offset2 = self.compute_offsets(t)
        if offset1 not in self.mem_class_x.keys():
            # no exemplar in memory yet, output uniform distr. over classes in
            # task t above, we check presence of first class for this task, we
            # should check them all
            out = torch.Tensor(ns, self.n_classes).fill_(-10e10)
            out[:, offset1: offset2].fill_(1.0 / max(task_classes, 1))
            if self.gpu:
                out = out.cuda()
            return out
        means = torch.ones(task_classes, nd) * float('inf')
        if self.gpu:
            means = means.cuda()
        for cc in range(offset1, offset2):
            means[cc -
                  offset1] =self.netforward(self.mem_class_x[cc]).data.mean(0)
        classpred = torch.LongTensor(ns)
        preds = self.netforward(x).data.clone()
        for ss in range(ns):
            dist = (means - preds[ss].expand(task_classes, nd)).norm(2, 1)
            _, ii = dist.min(0)
            ii = ii.squeeze()
            classpred[ss] = ii.item() + offset1

        out = torch.zeros(ns, self.n_classes)
        if self.gpu:
            out = out.cuda()
        for ss in range(ns):
            out[ss, classpred[ss]] = 1
        return out  # return 1-of-C code, ns x nc

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

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

        x_det = self._prepare_input(x)
        x = x.view(x.size(0), -1)
        self.net.train()
        if self.gpu: self.net.cuda()
        y_cls, y_det = self._unpack_labels(y)
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        signal_mask = (y_det == 1) & (y_cls >= 0)
        if not signal_mask.any():
            self.det_opt.zero_grad()
            det_logits, _ = self.net.forward_heads(x_det)
            det_loss = self.det_loss(det_logits, y_det.float())
            det_replay = self._sample_det_memory()
            if det_replay is not None:
                mem_x, mem_y = det_replay
                mem_det_logits, _ = self.net.forward_heads(mem_x)
                mem_loss = self.det_loss(mem_det_logits, mem_y.float())
                det_loss = 0.5 * (det_loss + mem_loss)
            det_loss = self.det_lambda * det_loss
            det_loss.backward()
            self.det_opt.step()
            return float(det_loss.item()), 0.0

        x = x[signal_mask]
        y = y_cls[signal_mask]

        tr_acc = []

        for pass_itr in range(self.glances):

            # only make changes like pushing to buffer once per batch and not for every glance
            if(pass_itr==0):
                self.examples_seen += x.size(0)
                samples_per_task = self._get_samples_per_task(t)
                assert samples_per_task > 0, 'Samples per task is <= 0'

                # if not last batch of task, store samples in memx/memy
                # Problem if batch_size <= samples_per_task
                if self.examples_seen < samples_per_task:
                    if self.memx is None:
                        self.memx = x.data.clone()
                        self.memy = y.data.clone()
                    else:
                        self.memx = torch.cat((self.memx, x.data.clone()))
                        self.memy = torch.cat((self.memy, y.data.clone()))

            self.net.zero_grad()
            offset1, offset2 = self.compute_offsets(t)
            logits = self.netforward(x)[:, offset1: offset2]
            targets = y - offset1
            preds = torch.argmax(logits, dim=1)
            tr_acc.append(macro_recall(preds, targets))
            loss = self.bce(logits, targets)
            # num_exemplars remains 0 unless final epoch is reached
            if self.num_exemplars > 0:
                # distillation
                for tt in range(t):
                    # first generate a minibatch with one example per class from
                    # previous tasks
                    task_classes = self.classes_per_task[tt]
                    inp_dist = torch.zeros(task_classes, x.size(1))
                    target_dist = torch.zeros(task_classes, self.n_feat)
                    offset1, offset2 = self.compute_offsets(tt)
                    if self.gpu:
                        inp_dist = inp_dist.cuda()
                        target_dist = target_dist.cuda()
                    for cc in range(task_classes):
                        indx = random.randint(0, len(self.mem_class_x[cc + offset1]) - 1)
                        inp_dist[cc] = self.mem_class_x[cc + offset1][indx].clone()
                        target_dist[cc] = self.mem_class_y[cc +
                                                           offset1][indx].clone()
                    # Add distillation loss
                    loss += self.reg * self.kl(
                        self.lsm(self.netforward(inp_dist)
                                 [:, offset1: offset2]),
                        self.sm(target_dist[:, offset1: offset2])) * task_classes
            # bprop and update
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            self.opt.step()

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!
        target = int(self.cfg.n_epochs * self._get_samples_per_task(t))
        if self.examples_seen >= target:  # not ==
            self.examples_seen = 0
            # self._rebuild_exemplars_for_task(t, x.device)

        # if self.examples_seen == self.cfg.n_epochs * self.samples_per_task:
        #     self.examples_seen = 0
            # get labels from previous task; we assume labels are consecutive
            offset1, offset2 = self.compute_offsets(t)
            if self.gpu:
                all_labs = torch.LongTensor(np.unique(self.memy.cpu().numpy()))
            else:
                all_labs = torch.LongTensor(np.unique(self.memy.numpy()))
            
            num_classes = all_labs.size(0)
            # if self.cfg.loader == 'class_incremental_loader':
            #     num_classes = all_labs.size(0)
            # else:
            #     num_classes = all_labs[offset1:offset2].size(0)
            
            print("num_classes", num_classes, "nc_per_task", self.nc_per_task,
                  offset1, offset2)
            current_task_classes = self.classes_per_task[t]
            assert num_classes == current_task_classes
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
                nd = num_classes # num classes in current task
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
            # print(len(self.mem_class_x[0]))

        avg_tr_acc = sum(tr_acc) / len(tr_acc) if tr_acc else 0.0
        self.det_opt.zero_grad()
        det_logits, _ = self.net.forward_heads(x_det)
        det_loss = self.det_loss(det_logits, y_det.float())
        det_replay = self._sample_det_memory()
        if det_replay is not None:
            mem_x, mem_y = det_replay
            mem_det_logits, _ = self.net.forward_heads(mem_x)
            mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            det_loss = 0.5 * (det_loss + mem_loss)
        det_loss = self.det_lambda * det_loss
        det_loss.backward()
        self.det_opt.step()
        return loss.item(), avg_tr_acc
