# An implementation of Experience Replay (ER) with reservoir sampling and without using tasks from Algorithm 4 of https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import warnings
import math

from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
warnings.filterwarnings("ignore")
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class ErAlgConfig:
    alpha_init: float = 1e-3
    lr: float = 1e-3
    opt_lr: float = 1e-1
    learn_lr: bool = False
    glances: int = 1
    memories: int = 5120
    replay_batch_size: int = 20
    grad_clip_norm: Optional[float] = 2.0
    second_order: bool = False
    meta_batches: int = 3

    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "ErAlgConfig":
        cfg = ErAlgConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(DetectionReplayMixin, nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        self.cfg = ErAlgConfig.from_args(args)

        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.opt_wt = optim.SGD(self._ll_params(), lr=self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(self.net.det_head.parameters(), lr=self.cfg.lr, momentum=0.9)

        if self.cfg.learn_lr:
            self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=self.cfg.opt_lr, momentum=0.9)    

        self.loss = CrossEntropyLoss()
        self.is_cifar = ((self.cfg.dataset == 'cifar100') or (self.cfg.dataset == 'tinyimagenet'))
        self.glances = self.cfg.glances
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.current_task = 0
        self.memories = self.cfg.memories
        self.batchSize = int(self.cfg.replay_batch_size)

        # allocate buffer
        self.M = []
        self.age = 0
        
        # handle gpus if specified
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
        # if self.is_cifar:
        #     self.nc_per_task = int(n_outputs / n_tasks)
        # else:
        #     self.nc_per_task = n_outputs


    def compute_offsets(self, task):
        return misc_utils.compute_offsets(task, self.classes_per_task)

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param
            
    def take_multitask_loss(self, bt, logits, y):
        loss = 0.0
        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def forward(self, x, t):
        output = self.net.forward(x)
        if True: #self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def getBatch(self, x, y, t):
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

        if len(self.M) > 0:
            order = [i for i in range(0,len(self.M))]
            osize = min(self.batchSize,len(self.M))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]
                x,y,t = self.M[k]
                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        for i in range(len(myi)):
            bxs.append(mxi[i])
            bys.append(myi[i])
            bts.append(mti[i])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        
        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()
 
        return bxs,bys,bts


    def observe(self, x, y, t):
        ### step through elements of x

        class_counts = getattr(self, "classes_per_task", None)
        noise_label = None
        if class_counts is not None:
            _, offset2 = misc_utils.compute_offsets(t, class_counts)
            noise_label = offset2 - 1
        y_cls, y_det = self._unpack_labels(
            y,
            noise_label=noise_label,
            use_detector_arch=bool(getattr(self, "det_enabled", False)),
        )
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        x_det = x
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
        xi = x.data.cpu().numpy()
        yi = y.data.cpu().numpy()

        if t != self.current_task:
           self.current_task = t

        if self.cfg.learn_lr:
            loss, tr_acc = self.la_ER(x, y, t)
        else:
            loss, tr_acc = self.ER(xi, yi, t)

        for i in range(0, x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i], yi[i], t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi[i], yi[i], t]

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

        return loss.item(), tr_acc

    def _batch_accuracy(self, bt, logits, labels):
        if len(bt) == 0:
            return 0.0
        preds_list = []
        target_list = []
        with torch.no_grad():
            for idx, task_idx in enumerate(bt):
                offset1, offset2 = self.compute_offsets(int(task_idx))
                preds = torch.argmax(logits[idx, offset1:offset2], dim=0)
                target = labels[idx] - offset1
                preds_list.append(preds.detach().cpu())
                target_list.append(target.detach().cpu())
        if not preds_list:
            return 0.0
        stacked_preds = torch.stack(preds_list).view(-1)
        stacked_targets = torch.stack(target_list).view(-1)
        return macro_recall(stacked_preds, stacked_targets)

    def ER(self, x, y, t):
        tr_acc = []
        for pass_itr in range(self.glances):

            self.net.zero_grad()
            
            # Draw batch from buffer:
            bx,by,bt = self.getBatch(x,y,t)

            bx = bx.squeeze()
            prediction = self.net.forward(bx)
            loss = self.take_multitask_loss(bt, prediction, by)
            tr_acc.append(self._batch_accuracy(bt, prediction, by))

            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            self.opt_wt.step()
        
        avg_tr_acc = sum(tr_acc)/len(tr_acc) if tr_acc else 0.0
        return loss, avg_tr_acc

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """

        # if self.is_cifar:
        #     offset1, offset2 = self.compute_offsets(t)            
        #     logits = self.net.forward(x, fast_weights)[:, :offset2]
        #     loss = self.loss(logits[:, offset1:offset2], y-offset1)
        # else:
        #     logits = self.net.forward(x, fast_weights)
        #     loss = self.loss(logits, y)

        offset1, offset2 = self.compute_offsets(t)            
        logits = self.net.forward(x, fast_weights)[:, :offset2]
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        if fast_weights is None:
            # fast_weights = self.net.parameters()
            fast_weights = list(self.net.parameters())

        graph_required = self.cfg.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required))

        for i in range(len(grads)):
            if self.cfg.grad_clip_norm:
                clip_val = self.cfg.grad_clip_norm
                grads[i] = torch.clamp(grads[i], min = -clip_val, max = clip_val)

        fast_weights = list(
            map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights, loss.item()


    def la_ER(self, x, y, t):
        """
        this ablation tests whether it suffices to just do the learning rate modulation
        guided by gradient alignment + clipping (that La-MAML does implciitly through autodiff)
        and use it with ER (therefore no meta-learning for the weights)

        """
        tr_acc = []
        for pass_itr in range(self.glances):
            
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            batch_sz = x.shape[0]
            n_batches = self.cfg.meta_batches
            rough_sz = math.ceil(batch_sz/n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)] 

            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            bx = bx.squeeze()
            
            for i in range(n_batches):

                batch_x = x[i*rough_sz : (i+1)*rough_sz]
                batch_y = y[i*rough_sz : (i+1)*rough_sz]

                # assuming labels for inner update are from the same 
                fast_weights, inner_loss = self.inner_update(batch_x, fast_weights, batch_y, t)

                prediction = self.net.forward(bx, fast_weights)
                meta_loss = self.take_multitask_loss(bt, prediction, by)
                meta_losses[i] += meta_loss

            # update alphas
            self.net.zero_grad()
            self.opt_lr.zero_grad()

            meta_loss = meta_losses[-1] #sum(meta_losses)/len(meta_losses)
            meta_loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.cfg.grad_clip_norm)
            
            # update the LRs (guided by meta-loss, but not the weights)
            self.opt_lr.step()

            # update weights
            self.net.zero_grad()

            # compute ER loss for network weights
            prediction = self.net.forward(bx)
            loss = self.take_multitask_loss(bt, prediction, by)
            tr_acc.append(self._batch_accuracy(bt, prediction, by))

            loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            # update weights with grad from simple ER loss 
            # and LRs obtained from meta-loss guided by old and new tasks
            for i,p in enumerate(self.net.parameters()):                                 
                p.data = p.data - (p.grad * nn.functional.relu(self.net.alpha_lr[i]))       
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        avg_tr_acc = sum(tr_acc)/len(tr_acc) if tr_acc else 0.0
        return loss, avg_tr_acc
