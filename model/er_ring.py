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
import numpy as np
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class ErRingConfig:
    memory_strength: float = 1.0
    lr: float = 1e-3
    n_memories: int = 2000
    replay_batch_size: int = 20
    inner_steps: int = 5

    batch_size: int = 128
    cuda: bool = True
    # temperature: float = 2.0
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "ErRingConfig":
        cfg = ErRingConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(DetectionReplayMixin, torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = ErRingConfig.from_args(args)
        self.reg = self.cfg.memory_strength
        # self.temp = self.cfg.temperature
        # setup network
        self.is_task_incremental = True
        self.net = ResNet1D(n_outputs, args)
        # setup optimizer
        self.lr = self.cfg.lr
        # if self.is_task_incremental:
        #    self.opt = torch.optim.Adam(self.net.parameters(), lr='self.lr)
        # else:
        self.opt = torch.optim.SGD(self._ll_params(), lr=self.lr, momentum=0.9)
        self.det_opt = torch.optim.SGD(
            self.net.det_head.parameters(), lr=self.lr, momentum=0.9
        )

        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
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
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
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
        self.n_memories = int(self.cfg.n_memories)
        self.task_memory_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.max_task_memories = max(self.task_memory_capacities, default=0)

        # Replay buffer stores canonical shape (2, 512) from _input_for_replay (2-channel or adapter output).
        seq_len = n_inputs // 2
        self.memx = torch.FloatTensor(
            n_tasks, self.max_task_memories, 2, seq_len
        ).fill_(0)
        self.memy = torch.LongTensor(n_tasks, self.max_task_memories).fill_(-1)
        self.mem_feat = torch.FloatTensor(
            n_tasks, self.max_task_memories, self.nc_per_task
        ).fill_(0)
        self.mem = {}
        if self.cfg.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()
        self.bsz = self.cfg.batch_size
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.cfg.cuda:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        # Use batchmean to align with KL math and silence PyTorch warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps

    def on_epoch_end(self):
        pass

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total replay budget across tasks.

        Args:
            total_memories: Total replay-buffer capacity.
            n_tasks: Number of tasks in the stream.

        Returns:
            Per-task capacities whose sum equals `total_memories`.
        """
        if n_tasks <= 0:
            return []
        base_capacity = total_memories // n_tasks
        remainder_capacity = total_memories % n_tasks
        return [
            base_capacity + (1 if task_index < remainder_capacity else 0)
            for task_index in range(n_tasks)
        ]

    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

    def forward(self, x, t, return_feat=False):
        output = self.net(x)

        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2) : self.n_outputs].data.fill_(-10e10)
        return output

    def memory_sampling(self, t):
        filled_counts = [int(self.task_mem_filled[i].item()) for i in range(t)]
        total = sum(filled_counts)
        if total == 0:
            return None
        sz = int(min(total, self.sz))
        flat_indices = np.random.choice(total, sz, replace=False)

        # map flat indices to task/sample indices
        t_idx_list = []
        s_idx_list = []
        cum = np.cumsum([0] + filled_counts)
        for fi in flat_indices:
            task_idx = max(i for i in range(len(cum) - 1) if cum[i] <= fi)
            sample_idx = fi - cum[task_idx]
            t_idx_list.append(task_idx)
            s_idx_list.append(sample_idx)

        t_idx = torch.tensor(t_idx_list, dtype=torch.long, device=self.memx.device)
        s_idx = torch.tensor(s_idx_list, dtype=torch.long, device=self.memx.device)

        offsets = torch.tensor(
            [self.compute_offsets(int(i)) for i in t_idx.tolist()],
            device=self.memx.device,
        )
        xx = self.memx[t_idx, s_idx]
        yy = self.memy[t_idx, s_idx] - offsets[:, 0]
        feat = self.mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        for j in range(mask.size(0)):
            cls_size = offsets[j][1] - offsets[j][0]
            mask[j, :cls_size] = torch.arange(
                offsets[j][0], offsets[j][1], device=self.memx.device
            )
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx, yy, feat, mask.long(), sizes

    def observe(self, x, y, t):
        # t = info[0]
        # idx = info[1]
        self.net.train()
        # class_counts = getattr(self, "classes_per_task", None)
        # noise_label = None
        # if class_counts is not None:
        #     _, offset2 = misc_utils.compute_offsets(t, class_counts)
        #     noise_label = offset2 - 1
        # y_cls, y_det = self._unpack_labels(
        #     y,
        #     noise_label=noise_label,
        #     use_detector_arch=bool(getattr(self, "det_enabled", False)),
        # )
        # if y_det is not None and self.det_memories > 0:
        #     self._update_det_memory(x, y_det)
        # x_det = x
        # signal_mask = (y_det == 1) & (y_cls >= 0)
        # if not signal_mask.any():
        #     if not getattr(self, "det_enabled", True):
        #         return 0.0, 0.0
        #     self.det_opt.zero_grad()
        #     det_logits, _ = self.net.forward_heads(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits, _ = self.net.forward_heads(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss = self.det_lambda * det_loss
        #     det_loss.backward()
        #     self.det_opt.step()
        #     return float(det_loss.item()), 0.0

        # x = x[signal_mask]
        # y = y_cls[signal_mask]
        task_capacity = int(self.task_memory_capacities[t])
        if task_capacity > 0:
            write_pointer = int(self.task_mem_ptr[t].item())
            bsz = y.data.size(0)
            endcnt = min(write_pointer + bsz, task_capacity)
            effbsz = endcnt - write_pointer
            if effbsz > 0:
                x_for_storage = self._input_for_replay(x)
                self.memx[t, write_pointer:endcnt].copy_(x_for_storage.data[:effbsz])
                self.memy[t, write_pointer:endcnt].copy_(y.data[:effbsz])
                filled_before_update = int(self.task_mem_filled[t].item())
                self.task_mem_filled[t] = min(
                    task_capacity, filled_before_update + effbsz
                )
            self.task_mem_ptr[t] = 0 if endcnt == task_capacity else endcnt

        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            # out = self.forward(self.memx[tt],tt, True)
            # self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t

        cls_tr_rec = []

        for _ in range(self.inner_steps):
            self.net.zero_grad()
            loss1 = torch.tensor(0.0).cuda()
            loss2 = torch.tensor(0.0).cuda()
            loss3 = torch.tensor(0.0).cuda()

            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x, t, True)
            logits = pred[:, offset1:offset2]
            targets = y - offset1
            if targets.min() < 0 or targets.max() >= logits.size(1):
                raise ValueError(
                    f"Target out of range for task {t}: "
                    f"min={int(targets.min())}, max={int(targets.max())}, "
                    f"class_count={logits.size(1)}, offset=({offset1},{offset2})"
                )
            preds = torch.argmax(logits, dim=1)
            cls_tr_rec.append(macro_recall(preds, targets))
            loss1 = classification_cross_entropy(
                logits, targets, class_weighted_ce=self.class_weighted_ce
            )
            if t > 0:
                sampled = self.memory_sampling(t)
                if sampled is not None:
                    xx, yy, target, mask, class_sizes = sampled
                    pred_ = self.net(xx)
                    pred = torch.gather(pred_, 1, mask)
                    for row, size in enumerate(class_sizes):
                        if size < pred.size(1):
                            pred[row, size:] = -1e9
                    if yy.min() < 0 or yy.max() >= pred.size(1):
                        raise ValueError(
                            f"Replay target out of range: min={int(yy.min())}, max={int(yy.max())}, "
                            f"class_count={pred.size(1)}, sizes={class_sizes.tolist()}"
                        )
                    loss2 += classification_cross_entropy(
                        pred, yy, class_weighted_ce=self.class_weighted_ce
                    )

            loss = loss1 + loss2
            loss.backward()
            self.opt.step()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        # if getattr(self, "det_enabled", True):
        #     self.det_opt.zero_grad()
        #     det_logits, _ = self.net.forward_heads(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits, _ = self.net.forward_heads(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss = self.det_lambda * det_loss
        #     det_loss.backward()
        #     self.det_opt.step()
        return loss.item(), avg_cls_tr_rec
