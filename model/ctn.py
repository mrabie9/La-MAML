# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

# from .common import ContextMLP, ContextNet18
# from .resnet import ResNet18 as ResNet18Full
from model.ctn_base import ContextNet18
from model.detection_replay import DetectionReplayMixin
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class CtnConfig:
    memory_strength: float = 0.5
    temperature: float = 5.0
    task_emb: int = 64
    lr: float = 0.01
    ctx_lr: float = 0.05
    n_memories: int = 50
    validation: float = 0.0
    replay_batch_size: int = 20
    inner_steps: int = 2
    n_meta: int = 2
    arch: str = "resnet1d"
    cuda: bool = True
    batch_size: int = 128
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "CtnConfig":
        cfg = CtnConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(DetectionReplayMixin, torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = CtnConfig.from_args(args)
        self.reg = self.cfg.memory_strength
        self.temp = self.cfg.temperature
        # setup network
        if self.cfg.arch == "resnet1d":
            # self.net = ResNet1D(n_outputs, args)
            use_iq_aug_features = bool(getattr(args, "use_iq_aug_features", False))
            iq_aug_scaling_mode = str(getattr(args, "data_scaling", "none"))
            iq_aug_feature_type = str(
                getattr(
                    args,
                    "iq_aug_feature_type",
                    getattr(args, "iq_aug_feature", "power"),
                )
            )
            in_channels = 3 if use_iq_aug_features else 2
            self.net = ContextNet18(
                n_outputs,
                in_channels=in_channels,
                n_tasks=n_tasks,
                task_emb=self.cfg.task_emb,
                use_iq_aug_features=use_iq_aug_features,
                iq_aug_scaling_mode=iq_aug_scaling_mode,
                iq_aug_feature_type=iq_aug_feature_type,
            )
        # self.net.define_task_lr_params(alpha_init=args.alpha_init)
        else:
            raise NotImplementedError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )

        self.is_task_incremental = True
        self.inner_lr = self.cfg.lr
        self.outer_lr = self.cfg.ctx_lr
        self.opt = torch.optim.SGD(
            self.net.parameters(), lr=self.outer_lr, momentum=0.9
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
        self.task_total_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.task_val_capacities = [
            int(task_capacity * self.cfg.validation)
            for task_capacity in self.task_total_capacities
        ]
        self.task_replay_capacities = [
            task_capacity - task_val
            for task_capacity, task_val in zip(
                self.task_total_capacities, self.task_val_capacities
            )
        ]
        self.max_task_replay_capacity = max(self.task_replay_capacities, default=0)
        self.max_task_val_capacity = max(self.task_val_capacities, default=0)

        # set up the semantic memory
        self.full_val = True  # avoid OOM when using too large memory

        # if 'cub' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val, 3, 224, 224)
        # elif 'mini' in args.data_file or 'core' in args.data_file:
        #     self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 84, 84)
        #     self.valx = torch.FloatTensor(n_tasks, self.n_val , 3, 84, 84)
        #     if self.n_memories > 75:
        #         self.full_val = False
        # else:
        self.memx = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, 2, n_inputs // 2
        )
        self.valx = torch.FloatTensor(
            n_tasks, self.max_task_val_capacity, 2, n_inputs // 2
        )

        self.memy = torch.LongTensor(n_tasks, self.max_task_replay_capacity)
        self.valy = torch.LongTensor(n_tasks, self.max_task_val_capacity)
        self.mem_feat = torch.FloatTensor(
            n_tasks, self.max_task_replay_capacity, self.nc_per_task
        )
        self.mem = {}
        if self.cfg.cuda:
            self.valx = self.valx.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.memy = self.memy.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)
            # self.valy.data.fill_(0)

        self.task_mem_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_mem_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_val_ptr = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.task_val_filled = torch.zeros(
            n_tasks, dtype=torch.long, device=self.memx.device
        )
        self.bsz = self.cfg.batch_size

        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        # Use batchmean to align with KL definition and silence PyTorch deprecation warning
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.samples_seen = 0
        self.sz = int(self.cfg.replay_batch_size)
        self.inner_steps = self.cfg.inner_steps
        self.n_meta = self.cfg.n_meta
        self.counter = 0

    def on_epoch_end(self):
        self.counter += 1
        pass

    def _build_task_memory_capacities(
        self, total_memories: int, n_tasks: int
    ) -> list[int]:
        """Split a total replay budget across tasks.

        Args:
            total_memories: Total replay capacity configured through `n_memories`.
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

    def forward(self, x, t, return_feat=False):
        output = self.net(x, t)

        if self.is_task_incremental:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2) : self.n_outputs].data.fill_(-10e10)
        return output

    def memory_sampling(self, t: int, valid: bool = False):
        """Sample replay or validation examples from buffered tasks.

        Args:
            t: Number of tasks to sample from, starting from task index `0`.
            valid: Whether to sample validation memory.

        Returns:
            Tuple of sampled tensors, or `None` when no memory exists.
        """
        if valid:
            filled_counts = [int(self.task_val_filled[i].item()) for i in range(t)]
        else:
            filled_counts = [int(self.task_mem_filled[i].item()) for i in range(t)]

        total_samples = sum(filled_counts)
        if total_samples <= 0:
            return None

        if valid and self.full_val:
            flat_sample_indices = np.arange(total_samples)
        else:
            sample_size = (
                min(total_samples, 64) if valid else int(min(total_samples, self.sz))
            )
            if sample_size <= 0:
                return None
            flat_sample_indices = np.random.choice(
                total_samples, sample_size, replace=False
            )

        cumulative_counts = np.cumsum([0] + filled_counts)
        task_indices_np = (
            np.searchsorted(cumulative_counts, flat_sample_indices, side="right") - 1
        )
        sample_indices_np = flat_sample_indices - cumulative_counts[task_indices_np]
        t_idx = torch.from_numpy(task_indices_np).to(
            device=self.memx.device, dtype=torch.long
        )
        s_idx = torch.from_numpy(sample_indices_np).to(
            device=self.memx.device, dtype=torch.long
        )

        offsets = torch.tensor(
            [self.compute_offsets(int(task_index)) for task_index in t_idx.tolist()],
            device=self.memx.device,
            dtype=torch.long,
        )
        if valid:
            xx = self.valx[t_idx, s_idx]
            yy = self.valy[t_idx, s_idx] - offsets[:, 0]
            feat = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        else:
            xx = self.memx[t_idx, s_idx]
            yy = self.memy[t_idx, s_idx] - offsets[:, 0]
            feat = self.mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task, device=self.memx.device)
        for row_index in range(mask.size(0)):
            class_size = offsets[row_index][1] - offsets[row_index][0]
            mask[row_index, :class_size] = torch.arange(
                offsets[row_index][0],
                offsets[row_index][1],
                device=self.memx.device,
            )
        sizes = (offsets[:, 1] - offsets[:, 0]).long()
        return xx, yy, feat, mask.long(), t_idx.tolist(), sizes

    def observe(self, x, y, t):
        class_counts = getattr(self, "classes_per_task", None)
        noise_label = None
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
        #     det_logits = self.net.forward_det_agnostic(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits = self.net.forward_det_agnostic(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     self.zero_grad()
        #     grads = torch.autograd.grad(
        #         det_loss,
        #         self.net.base_param(),
        #         create_graph=False,
        #         allow_unused=True,
        #     )
        #     for param, grad in zip(self.net.base_param(), grads):
        #         if grad is None:
        #             continue
        #         with torch.no_grad():
        #             param.add_(grad, alpha=-self.inner_lr)
        #     return float(det_loss.item()), 0.0
        # x = x[signal_mask]
        # y = y_cls[signal_mask]
        x_train = self._canonicalize_input(x, detach=False)
        x_for_storage = self._input_for_replay(x)

        # if task has changed, run model on val set of previous task to get soft targets
        if t != self.current_task:
            tt = self.current_task
            previous_task_filled = int(self.task_mem_filled[tt].item())
            if previous_task_filled > 0:
                offset1, offset2 = self.compute_offsets(tt)
                out = self.forward(self.memx[tt, :previous_task_filled], tt, True)
                cls_size = int(offset2 - offset1)
                feat = self.mem_feat[tt, :previous_task_filled]
                feat.zero_()
                feat[:, :cls_size] = F.softmax(
                    out[:, offset1:offset2] / self.temp, dim=1
                ).data.clone()  # store soft targets
            self.current_task = t
            self.memy[t] = 0

        # maintain validation set (store adapted input in val buffer)
        n_val_taken = 0
        n_rotated_in = 0
        task_val_capacity = int(self.task_val_capacities[t])
        if task_val_capacity > 0 and x_train.size(0) > 0:
            n_val_taken = 1
            incoming_val_y = y[0]
            x_train = x_train[1:]
            y = y[1:]
            val_write_pointer = int(self.task_val_ptr[t].item())
            val_filled = int(self.task_val_filled[t].item())
            # Only rotate in when overwriting a slot that has valid data (buffer full)
            if val_filled >= task_val_capacity:
                n_rotated_in = 1
                x_train = torch.cat(
                    [x_train, self.valx[t, val_write_pointer].unsqueeze(0)]
                )
                y = torch.cat([y, self.valy[t, val_write_pointer].unsqueeze(0)])
            self.valx[t, val_write_pointer].copy_(x_for_storage[0])
            self.valy[t, val_write_pointer].copy_(incoming_val_y)
            filled_val_before_update = int(self.task_val_filled[t].item())
            self.task_val_filled[t] = min(
                task_val_capacity, filled_val_before_update + 1
            )
            self.task_val_ptr[t] = (
                0
                if (val_write_pointer + 1) == task_val_capacity
                else (val_write_pointer + 1)
            )
            if x_train.size(0) == 0:
                x_train = x_for_storage[0].unsqueeze(0)
                y = incoming_val_y.unsqueeze(0)
        # memory set: only write "new" samples to replay; rotated-in sample is already in val buffer
        self.net.train()
        task_replay_capacity = int(self.task_replay_capacities[t])
        if task_replay_capacity > 0 and y.data.size(0) > 0:
            replay_write_pointer = int(self.task_mem_ptr[t].item())
            batch_size = y.data.size(0)
            n_new = batch_size - n_rotated_in
            endcnt = min(replay_write_pointer + n_new, task_replay_capacity)
            effbsz = endcnt - replay_write_pointer
            if effbsz > 0:
                replay_start = n_val_taken
                self.memx[t, replay_write_pointer:endcnt].copy_(
                    x_for_storage[replay_start : replay_start + effbsz]
                )
                self.memy[t, replay_write_pointer:endcnt].copy_(y.data[:effbsz])
                filled_mem_before_update = int(self.task_mem_filled[t].item())
                self.task_mem_filled[t] = min(
                    task_replay_capacity, filled_mem_before_update + effbsz
                )
            self.task_mem_ptr[t] = 0 if endcnt == task_replay_capacity else endcnt

        # if getattr(self, "det_enabled", True):
        #     det_logits = self.net.forward_det_agnostic(x_det)
        #     det_loss = self.det_loss(det_logits, y_det.float())
        #     det_replay = self._sample_det_memory()
        #     if det_replay is not None:
        #         mem_x, mem_y = det_replay
        #         mem_det_logits = self.net.forward_det_agnostic(mem_x)
        #         mem_loss = self.det_loss(mem_det_logits, mem_y.float())
        #         det_loss = 0.5 * (det_loss + mem_loss)
        #     det_loss_value = det_loss.detach()
        #     det_loss = self.det_lambda * det_loss
        #     det_grads = torch.autograd.grad(
        #         det_loss,
        #         self.net.base_param(),
        #         create_graph=False,
        #         allow_unused=True,
        #     )
        #     for param, grad in zip(self.net.base_param(), det_grads):
        #         if grad is None:
        #             continue
        #         with torch.no_grad():
        #             param.add_(grad, alpha=-self.inner_lr)
        # else:
        if True:
            det_loss_value = torch.zeros((), device=x_train.device, dtype=torch.float32)

        # Keep a detached canonical source and rebuild the train tensor per meta step
        # to avoid reusing a freed autograd graph across meta iterations.
        x_train_source = x_train.detach()
        self.zero_grad()
        cls_tr_rec = []
        context_parameters = list(self.net.context_param())
        for _ in range(self.n_meta):
            x_train = self._canonicalize_input(x_train_source, detach=False)
            loss1 = torch.tensor(0.0, device=x.device)

            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x_train, t)
            logits = pred[:, offset1:offset2]
            targets = y - offset1
            preds = torch.argmax(logits, dim=1)
            local_noise_label = None
            if (not getattr(self, "det_enabled", False)) and noise_label is not None:
                local_noise_label = noise_label - offset1
            if local_noise_label is None:
                signal_mask_for_metric = torch.ones_like(targets, dtype=torch.bool)
            else:
                signal_mask_for_metric = targets != local_noise_label
            if signal_mask_for_metric.any():
                cls_tr_rec.append(
                    macro_recall(
                        preds[signal_mask_for_metric], targets[signal_mask_for_metric]
                    )
                )
            else:
                cls_tr_rec.append(0.0)

            loss1 = classification_cross_entropy(
                logits, targets, class_weighted_ce=self.class_weighted_ce
            )
            # tt = t + 1
            for i in range(self.inner_steps):
                loss2 = torch.tensor(0.0, device=x.device)
                loss3 = torch.tensor(0.0, device=x.device)
                if t > 0:
                    sampled = self.memory_sampling(t)
                    if sampled is not None:
                        xx, yy, feat, mask, list_t, class_sizes = sampled
                        pred_ = self.net(xx, list_t)
                        pred = torch.gather(pred_, 1, mask)
                        for row, size in enumerate(class_sizes):
                            if size < pred.size(1):
                                pred[row, size:] = -1e9
                        loss2 = classification_cross_entropy(
                            pred, yy, class_weighted_ce=self.class_weighted_ce
                        )
                        loss3 = self.reg * self.kl(
                            F.log_softmax(pred / self.temp, dim=1), feat
                        )
                    loss = (
                        self.cls_lambda * loss1
                        + self.det_lambda * det_loss_value
                        + loss2
                        + loss3
                    )
                else:
                    loss = self.cls_lambda * loss1 + self.det_lambda * det_loss_value

                grads = torch.autograd.grad(
                    loss,
                    self.net.base_param(),
                    create_graph=False,
                    allow_unused=True,
                )

                # SGD update only the BASE NETWORK
                for param, grad in zip(self.net.base_param(), grads):
                    if grad is None:
                        continue
                    with torch.no_grad():
                        param.add_(grad, alpha=-self.inner_lr)

            sampled_validation = self.memory_sampling(t + 1, valid=True)
            if sampled_validation is None:
                outer_loss = classification_cross_entropy(
                    logits, targets, class_weighted_ce=self.class_weighted_ce
                )
            else:
                xval, yval, feat, mask, list_t, class_sizes_val = sampled_validation
                pred_ = self.net(xval, list_t)
                pred = torch.gather(pred_, 1, mask)
                for row, size in enumerate(class_sizes_val):
                    if size < pred.size(1):
                        pred[row, size:] = -1e9
                outer_loss = classification_cross_entropy(
                    pred, yval, class_weighted_ce=self.class_weighted_ce
                )
            outer_grad = torch.autograd.grad(
                outer_loss,
                context_parameters,
                create_graph=False,
                allow_unused=True,
            )

            self.opt.zero_grad()
            for param, grad in zip(context_parameters, outer_grad):
                if grad is None:
                    continue
                param.grad = grad.detach().clamp(-1, 1)
            self.opt.step()
            # SGD update the CONTROLLER
            self.zero_grad()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss.item(), avg_cls_tr_rec
