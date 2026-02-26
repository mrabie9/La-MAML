# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class BclDualConfig:
    lr: float = 1e-3
    beta: float = 1.0
    memory_strength: float = 1.0
    temperature: float = 5.0
    n_memories: int = 2000
    inner_steps: int = 5
    n_meta: int = 5

    cuda: bool = True
    replay_batch_size: int = 20
    det_lambda: float = 10.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "BclDualConfig":
        cfg = BclDualConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

class Net(DetectionReplayMixin, torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = BclDualConfig.from_args(args)
        self.n_tasks = n_tasks
        self.reg = self.cfg.memory_strength
        self.temp = self.cfg.temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)

        # setup optimizer
        self.inner_lr = self.cfg.lr
        self.beta= self.cfg.beta
        #self.outer_opt = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.net.parameters(), lr=self.inner_lr, momentum=0.9)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

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
        
        self.n_val = int(self.n_memories * 0.2)
        self.n_memories -= self.n_val
        
        self.memx = torch.FloatTensor(n_tasks, self.n_memories, 2, n_inputs//2)
        self.valx = torch.FloatTensor(n_tasks, self.n_val, 2, n_inputs//2)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task).fill_(0)
        self.valy = torch.LongTensor(n_tasks, self.n_val)
        self.mem = {}
        if self.cfg.cuda:
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
        # Use batchmean to match KL definition and remove PyTorch warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.glances = self.cfg.inner_steps
        self.n_meta = self.cfg.n_meta
        self.count = 0
        self.val_count = 0
        self.adapt_ = False #args.adapt
        self.adapt_lr = self.cfg.lr
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
            opt = torch.optim.SGD(model.parameters(), self.adapt_lr, momentum=0.9)
            train = torch.utils.data.TensorDataset(xx, yy)
            loader = DataLoader(train, batch_size = self.bsz, shuffle = True, num_workers =0)
            for _ in range(self.glances):
                model.zero_grad()
                pred = model.forward(xx)
                loss = self.bce(pred, yy)
                loss.backward()
                opt.step()
            self.models[t] = model


    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

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
            sz = int(min(t*self.n_val, self.sz))
            idx = np.random.choice(t* self.n_val,sz, False)
            self.valid_id = idx.tolist()
            t_idx = torch.from_numpy(idx // self.n_val)
            s_idx = torch.from_numpy( idx % self.n_val)
        else:
            mem_x = self.memx[:t,:]
            mem_y = self.memy[:t,:]
            mem_feat = self.mem_feat[:t,:]
            sz = int(min(self.n_memories, self.sz))
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
            cls_size = offsets[j][1] - offsets[j][0]
            mask[j, :cls_size] = torch.arange(offsets[j][0], offsets[j][1])
        mask = mask.long().cuda()
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx,yy, feat , mask, t_idx.tolist(), sizes
    def observe(self, x, y, t):
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
        if self.current_task is None:
            self.current_task = t
        if self.current_task == t and self.mem_cnt == 0 and y_det is not None:
            det_mean = float(y_det.float().mean().item())
            if det_mean >= 0.99:
                print("[WARN][BCL_Dual] y_det mean ~1.0; detection may be missing negatives.")
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        x_det = x
        signal_mask = (y_det == 1) & (y_cls >= 0)
        if not signal_mask.any():
            det_logits, _ = self.net.forward_heads(x_det)
            det_loss = self.det_loss(det_logits, y_det.float())
            det_replay = self._sample_det_memory()
            if det_replay is not None:
                mem_x, mem_y = det_replay
                mem_det_logits, _ = self.net.forward_heads(mem_x)
                mem_loss = self.det_loss(mem_det_logits, mem_y.float())
                det_loss = 0.5 * (det_loss + mem_loss)
            self.zero_grad()
            det_loss.backward()
            self.inner_opt.step()
            return float(det_loss.item()), 0.0
        x = x[signal_mask]
        y = y_cls[signal_mask]
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            cls_size = int(offset2 - offset1)
            feat = self.mem_feat[tt]
            feat.zero_()
            feat[:, :cls_size] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1).data.clone()
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
        tr_acc = []
        for _ in range(self.n_meta):
            weights_before = deepcopy(self.net.state_dict())
            offset1, offset2 = self.compute_offsets(t)
            for i in range(self.glances):
                pred = self.forward(x,t)
                logits = pred[:, offset1:offset2]
                targets = y - offset1
                preds = torch.argmax(logits, dim=1)
                tr_acc.append(macro_recall(preds, targets))
                loss1 = self.bce(logits, targets)
                det_logits, _ = self.net.forward_heads(x_det)
                det_loss = self.det_loss(det_logits, y_det.float())
                det_replay = self._sample_det_memory()
                if det_replay is not None:
                    mem_x, mem_y = det_replay
                    mem_det_logits, _ = self.net.forward_heads(mem_x)
                    mem_loss = self.det_loss(mem_det_logits, mem_y.float())
                    det_loss = 0.5 * (det_loss + mem_loss)
                if t > 0:
                    xx, yy, feat, mask, list_t, class_sizes = self.memory_sampling(t)
                    pred_ = self.net(xx)
                    pred = torch.gather(pred_, 1, mask)
                    for row, size in enumerate(class_sizes):
                        if size < pred.size(1):
                            pred[row, size:] = -1e9
                    loss2 = self.bce(pred, yy)
                    loss3 = self.reg * self.kl(F.log_softmax(pred / self.temp, dim = 1), feat)
                    loss = (self.cls_lambda * loss1
                            + self.det_lambda * det_loss
                            + loss2 + loss3)
                else:
                    loss = self.cls_lambda * loss1 + self.det_lambda * det_loss
                loss.backward()
                self.inner_opt.step()
            xval, yval, _, mask_val, list_t, class_sizes_val = self.memory_sampling(tt, valid = True)  
            pred_ = self.net(xval)
            pred = torch.gather(pred_, 1, mask_val)
            for row, size in enumerate(class_sizes_val):
                if size < pred.size(1):
                    pred[row, size:] = -1e9
            outer_loss = self.bce(pred, yval)
            outer_loss.backward()                    
            self.inner_opt.step()
            self.zero_grad()
            weights_after = self.net.state_dict()
            new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before.keys()}
            self.net.load_state_dict(new_params)
        avg_tr_acc = sum(tr_acc) / len(tr_acc) if tr_acc else 0.0
        return outer_loss.item(), avg_tr_acc
