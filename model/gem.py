# TODO: update mmemory buffer to store IQ values together

### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory. 
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

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
import quadprog

from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class GemConfig:
    memory_strength: float = 0.0 # lambda in the paper
    glances: int = 1
    lr: float = 1e-3
    n_memories: int = 0
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    cuda: bool = True
    alpha_init: float = 1e-3
    grad_clip_norm: Optional[float] = 100.0
    input_channels: int = 2
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "GemConfig":
        cfg = GemConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    return misc_utils.compute_offsets(task, nc_per_task)


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


class Net(DetectionReplayMixin, nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = GemConfig.from_args(args)
        self.margin = self.cfg.memory_strength
        self.is_cifar = (
            (self.cfg.dataset == 'cifar100') or (self.cfg.dataset == 'tinyimagenet')
        )

        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)
        self.netforward = self.net.forward
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.glances = self.cfg.glances
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(self.net.det_head.parameters(), self.cfg.lr, momentum=0.9)

        self.n_memories = int(self.cfg.n_memories)
        self.task_memory_capacities = self._build_task_memory_capacities(
            self.n_memories,
            n_tasks,
        )
        self.max_task_memories = max(self.task_memory_capacities, default=0)
        self.gpu = self.cfg.cuda

        # --- Episodic memory allocation ---
        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            # (task, mem, C=2, L)
            self.memory_data = torch.FloatTensor(n_tasks, self.max_task_memories, 2, self.seq_len)
        else:
            # (task, mem, F)
            self.memory_data = torch.FloatTensor(n_tasks, self.max_task_memories, n_inputs)

        self.memory_labs = torch.LongTensor(n_tasks, self.max_task_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # track how many exemplars each task has actually written
        self.task_mem_filled = torch.zeros(n_tasks, dtype=torch.long)
        self.task_mem_ptr = torch.zeros(n_tasks, dtype=torch.long)
        if self.gpu:
            self.task_mem_filled = self.task_mem_filled.cuda()
            self.task_mem_ptr = self.task_mem_ptr.cuda()

        # --- GEM gradient buffers ---
        self.grad_dims = [p.data.numel() for p in self._ll_params()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # --- counters / bookkeeping ---
        self.observed_tasks = []
        self.old_task = -1
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        if self.gpu:
            self.cuda()

    def _build_task_memory_capacities(self, total_memories: int, n_tasks: int) -> list[int]:
        """Split a total memory budget across tasks.

        Args:
            total_memories: Total replay-buffer capacity requested by config.
            n_tasks: Number of tasks in the stream.

        Returns:
            A list of per-task capacities whose sum equals `total_memories`.
        """
        if n_tasks <= 0:
            return []
        base_capacity = total_memories // n_tasks
        remainder_capacity = total_memories % n_tasks
        return [
            base_capacity + (1 if task_index < remainder_capacity else 0)
            for task_index in range(n_tasks)
        ]

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

    def _adapt_for_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Convert IQ inputs to the canonical replay-memory shape.

        Args:
            x: Input batch in one of the supported IQ layouts.

        Returns:
            Tensor with shape ``(B, 2, L_memory)`` where ``L_memory`` matches
            the model replay buffer sequence length.
        """
        adapted_x = x
        if adapted_x.dim() == 4 and adapted_x.size(1) == 3 and adapted_x.size(2) == 2:
            adapted_x = self.net.model.input_adapter(adapted_x)
        elif adapted_x.dim() == 3 and adapted_x.size(1) == 3:
            if adapted_x.size(2) % 2 != 0:
                raise ValueError(
                    f"Expected even length for 3-ADC IQ input; got shape {tuple(adapted_x.shape)}."
                )
            sequence_length = adapted_x.size(2) // 2
            adapted_x = adapted_x.view(adapted_x.size(0), 3, 2, sequence_length)
            adapted_x = self.net.model.input_adapter(adapted_x)
        else:
            adapted_x = self._ensure_iq_shape(adapted_x)

        if adapted_x.size(1) != 2:
            raise ValueError(
                f"Replay memory expects 2 IQ channels; got shape {tuple(adapted_x.shape)}."
            )
        if adapted_x.size(2) > self.seq_len:
            adapted_x = adapted_x[:, :, : self.seq_len]
        elif adapted_x.size(2) < self.seq_len:
            pad_amount = self.seq_len - adapted_x.size(2)
            adapted_x = torch.nn.functional.pad(adapted_x, (0, pad_amount))
        return adapted_x

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

    def forward(self, x, t):
        if self.cfg.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x)  # (B, 2, L)

        output = self.netforward(x)

        # Task-incremental masking: always restrict logits to the task slice.
        offset1, offset2 = compute_offsets(t, self.classes_per_task, self.is_cifar)
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
            if not getattr(self, "det_enabled", True):
                return 0.0, 0.0
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

        # track tasks
        if t != self.old_task:
            if t not in self.observed_tasks:
                self.observed_tasks.append(t)
            self.old_task = t

        tr_acc = []

        for pass_itr in range(self.glances):
            # push current batch once per batch (not each glance)
            if pass_itr == 0:
                task_capacity = int(self.task_memory_capacities[t])
                if task_capacity > 0:
                    write_pointer = int(self.task_mem_ptr[t].item())
                    bsz = y.data.size(0)
                    endcnt = min(write_pointer + bsz, task_capacity)
                    effbsz = endcnt - write_pointer
                    # Store adapter output (e.g. 3-ADC -> 2-channel) so replay matches forward
                    mem_x = self._input_for_replay(x.data[:effbsz])
                    self.memory_data[t, write_pointer:endcnt].copy_(mem_x)

                    if bsz == 1:
                        self.memory_labs[t, write_pointer] = y.data[0]
                    else:
                        self.memory_labs[t, write_pointer:endcnt].copy_(y.data[:effbsz])

                    if effbsz > 0:
                        filled_before_update = int(self.task_mem_filled[t].item())
                        self.task_mem_filled[t] = min(task_capacity, filled_before_update + effbsz)
                    self.task_mem_ptr[t] = 0 if endcnt == task_capacity else endcnt

            # gradients on past tasks (replay)
            if len(self.observed_tasks) > 1:
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    past_task = self.observed_tasks[tt]
                    offset1, offset2 = compute_offsets(past_task, self.classes_per_task, self.is_cifar)
                    filled = int(self.task_mem_filled[past_task].item())
                    if filled == 0:
                        continue  # nothing stored for this task yet

                    # replay batch (shape already in memory)
                    mem_x = Variable(self.memory_data[past_task, :filled])          # (mem, F) or (mem, 2, L)
                    mem_y = Variable(self.memory_labs[past_task, :filled])          # (mem,)

                    logits_replay = self.forward(mem_x, past_task)[:, offset1:offset2]
                    targets_replay = mem_y - offset1
                    if targets_replay.min() < 0 or targets_replay.max() >= logits_replay.size(1):
                        raise ValueError(
                            f"GEM replay target out of range for task {past_task}: "
                            f"min={int(targets_replay.min())}, max={int(targets_replay.max())}, "
                            f"classes={logits_replay.size(1)}, offset=({offset1},{offset2})"
                        )
                    ptloss = self.ce(logits_replay, targets_replay)
                    ptloss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)
                    store_grad(self._ll_params, self.grads, self.grad_dims, past_task)

            # current batch
            self.zero_grad()
            offset1, offset2 = compute_offsets(t, self.classes_per_task, self.is_cifar)
            logits = self.forward(x, t)[:, offset1:offset2]
            targets = y - offset1
            if targets.min() < 0 or targets.max() >= logits.size(1):
                raise ValueError(
                    f"GEM target out of range for task {t}: "
                    f"min={int(targets.min())}, max={int(targets.max())}, classes={logits.size(1)}, "
                    f"offset=({offset1},{offset2})"
                )
            preds = torch.argmax(logits, dim=1)
            tr_acc.append(macro_recall(preds, targets))
            loss = self.ce(logits, targets)
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            # GEM projection if needed
            if len(self.observed_tasks) > 1:
                store_grad(self._ll_params, self.grads, self.grad_dims, t)
                device = torch.device("cuda") if self.gpu else torch.device("cpu")
                indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long, device=device)
                dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    overwrite_grad(self._ll_params, self.grads[:, t], self.grad_dims)

            self.opt.step()
        if getattr(self, "det_enabled", True):
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
        avg_tr_acc = sum(tr_acc) / len(tr_acc) if tr_acc else 0.0
        return loss.item(), avg_tr_acc
