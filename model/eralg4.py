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
from random import shuffle
import warnings
import math

from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

warnings.filterwarnings("ignore")


@dataclass
class ErAlgConfig:
    alpha_init: float = 1e-3
    lr: float = 1e-3
    opt_lr: float = 1e-1
    learn_lr: bool = False
    inner_steps: int = 1
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
    memory_loss_lambda: float = 1.0

    @staticmethod
    def from_args(args: object) -> "ErAlgConfig":
        cfg = ErAlgConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(DetectionReplayMixin, nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()

        self.cfg = ErAlgConfig.from_args(args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.opt_wt = optim.SGD(self._ll_params(), lr=self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(
            self.net.det_head.parameters(), lr=self.cfg.lr, momentum=0.9
        )

        if self.cfg.learn_lr:
            self.opt_lr = torch.optim.SGD(
                list(self.net.alpha_lr.parameters()), lr=self.cfg.opt_lr, momentum=0.9
            )

        self.is_cifar = (self.cfg.dataset == "cifar100") or (
            self.cfg.dataset == "tinyimagenet"
        )
        self.inner_steps = self.cfg.inner_steps
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self.memory_loss_lambda = float(self.cfg.memory_loss_lambda)
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
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.noise_label: int | None = noise_label_from_args(args)
        self.incremental_loader_name = getattr(args, "loader", None)
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
        # Global (unmasked) cross-entropy over the full ``n_outputs`` vector.
        # ``bt`` is unused for the loss itself (each row already carries its
        # global label); it is retained for signature parity with callers.
        if logits.size(0) == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        # The per-sample loop this replaces fed single-element batches through
        # ``classification_cross_entropy``, where inverse-frequency weights
        # collapse to 1.0; that is plain (unweighted) mean CE, which a single
        # batched call reproduces exactly while avoiding the Python loop.
        return classification_cross_entropy(
            logits,
            y.long(),
            class_weighted_ce=False,
        )

    def forward(self, x, t, *, cil_all_seen_upto_task=None):
        output = self.net.forward(x)
        if True:  # self.is_cifar:
            output = misc_utils.apply_task_incremental_logit_mask(
                output,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=cil_all_seen_upto_task,
                global_noise_label=self.noise_label,
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return output

    def getBatch(self, x, y, t):
        if x is not None:
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int) * t
        else:
            mxi = np.empty(shape=(0, 0))
            myi = np.empty(shape=(0, 0))
            mti = np.empty(shape=(0, 0))

        replay_x = []
        replay_y = []
        replay_t = []
        current_x = []
        current_y = []
        current_t = []

        if len(self.M) > 0:
            order = [i for i in range(0, len(self.M))]
            osize = min(self.batchSize, len(self.M))
            for j in range(0, osize):
                shuffle(order)
                k = order[j]
                x, y, t = self.M[k]
                xi = np.array(x)
                yi_scalar = int(torch.as_tensor(y).long().flatten()[0].item())
                ti = np.array(t)
                if self.noise_label is not None and yi_scalar == self.noise_label:
                    continue

                replay_x.append(xi)
                replay_y.append(yi_scalar)
                replay_t.append(ti)

        for i in range(len(myi)):
            current_x.append(mxi[i])
            current_y.append(myi[i])
            current_t.append(mti[i])

        bxs = replay_x + current_x
        bys = replay_y + current_y
        bts = replay_t + current_t
        replay_count = len(replay_x)

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)

        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs, bys, bts, replay_count

    def _weighted_multitask_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tasks: torch.Tensor,
        replay_count: int,
    ) -> torch.Tensor:
        replay_count = max(0, min(int(replay_count), logits.size(0)))
        replay_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if replay_count > 0:
            replay_loss = self.take_multitask_loss(
                tasks[:replay_count], logits[:replay_count], labels[:replay_count]
            )
        current_loss = self.take_multitask_loss(
            tasks[replay_count:], logits[replay_count:], labels[replay_count:]
        )
        return current_loss + (self.memory_loss_lambda * replay_loss)

    def observe(self, x, y, t):
        ### step through elements of x

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
        # Detached canonical (B, 2, L) tensor for reservoir storage only. The
        # current minibatch is trained on the *live* graph below (adapter
        # differentiable), so storage no longer feeds the training forward.
        x_for_storage = self._input_for_replay(x)
        y_work = unpack_y_to_class_labels(y).long()

        if t != self.current_task:
            self.current_task = t

        if self.cfg.learn_lr:
            # Keep a detached leaf copy so each inner/meta step can rebuild a
            # fresh canonicalized graph for 3-channel adapter inputs.
            raw_x_train = x.detach().requires_grad_(True)
            loss, cls_tr_rec, metric_logits = self.la_ER(raw_x_train, y, t)
        else:
            loss, cls_tr_rec, metric_logits = self.ER(x, y_work, t)

        # Reservoir-sampling memory update. Store detached canonical tensors on
        # CPU so the buffer holds pre-adapted (2, L) IQ rows (no grad through the
        # adapter for replayed samples is fine).
        x_store = x_for_storage.detach().cpu()
        y_store = y_work.detach().cpu()
        for i in range(0, x.size()[0]):
            self.age += 1
            if len(self.M) < self.memories:
                self.M.append([x_store[i], y_store[i], t])

            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [x_store[i], y_store[i], t]

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

        return loss.item(), cls_tr_rec, metric_logits

    def _batch_accuracy(self, bt, logits, labels):
        if len(bt) == 0:
            return 0.0
        preds_list = []
        target_list = []
        with torch.no_grad():
            for idx, task_idx in enumerate(bt):
                offset1, offset2 = self.compute_offsets(int(task_idx))
                y_g = int(labels[idx].item())
                if self.noise_label is not None and y_g == self.noise_label:
                    continue
                preds = torch.argmax(logits[idx, offset1:offset2], dim=0)
                target = labels[idx] - offset1
                preds_list.append(preds.detach().cpu())
                target_list.append(target.detach().cpu())
        if not preds_list:
            return 0.0
        stacked_preds = torch.stack(preds_list).view(-1)
        stacked_targets = torch.stack(target_list).view(-1)
        return macro_recall(stacked_preds, stacked_targets)

    def _sample_replay(self, device):
        """Sample a replay minibatch from the reservoir buffer ``M``.

        Returns pre-canonicalized ``(N, 2, L)`` GPU tensors plus their global
        labels and task ids, or ``None`` when no eligible samples exist. No
        gradient flows through the adapter for replayed (pre-adapted) rows.
        """
        if len(self.M) == 0:
            return None
        osize = min(self.batchSize, len(self.M))
        # The original loop reshuffled the full index list once per draw and took
        # position ``j``; that is uniform sampling-with-replacement of ``osize``
        # indices. ``random.choices`` reproduces it without the O(N * osize)
        # Python shuffling that dominated the replay hot path.
        indices = random.choices(range(len(self.M)), k=osize)
        replay_x = []
        replay_y = []
        replay_t = []
        for k in indices:
            xi, yi, ti = self.M[k]
            yi_scalar = int(torch.as_tensor(yi).long().flatten()[0].item())
            if self.noise_label is not None and yi_scalar == self.noise_label:
                continue
            replay_x.append(torch.as_tensor(xi))
            replay_y.append(yi_scalar)
            replay_t.append(int(ti))
        if not replay_x:
            return None
        bx = torch.stack(replay_x).float().to(device, non_blocking=True)
        by = torch.tensor(replay_y, dtype=torch.long, device=device)
        bt = torch.tensor(replay_t, dtype=torch.long, device=device)
        return bx, by, bt

    def ER(self, x, y, t):
        """Single training step per inner step on the live current minibatch.

        ``x`` is the raw current batch (3-ADC/4D or canonical IQ) and flows
        through the adapter with ``detach=False`` via ``net.forward``, so one
        ``loss.backward()`` + ``opt_wt.step()`` updates the backbone and the
        input adapter together. Replay rows are pre-canonicalized tensors drawn
        from the reservoir buffer (no adapter grad for old samples).
        """
        cls_tr_rec = []
        metric_logits = None
        current_t = torch.full((x.size(0),), int(t), dtype=torch.long, device=x.device)
        for pass_itr in range(self.inner_steps):

            self.net.zero_grad()

            # Current minibatch: live forward through the adapter. Unmasked
            # logits use global CE targets indexing the full ``n_outputs``
            # vector (including any shared noise class).
            current_logits = self.net.forward(x)
            current_loss = self.take_multitask_loss(current_t, current_logits, y)

            # Replay minibatch: pre-canonicalized rows from the buffer.
            replay = self._sample_replay(x.device)
            if replay is not None:
                replay_x, replay_y, replay_t = replay
                replay_logits = self.net.forward(replay_x)
                replay_loss = self.take_multitask_loss(
                    replay_t, replay_logits, replay_y
                )
            else:
                replay_loss = torch.zeros(
                    (), device=current_logits.device, dtype=current_logits.dtype
                )

            loss = current_loss + (self.memory_loss_lambda * replay_loss)

            # Progress-bar metric: masked task-incremental logits on the current
            # task, matching er_ring / icarl.
            masked_logits = misc_utils.apply_task_incremental_logit_mask(
                current_logits,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=t,
                global_noise_label=self.noise_label,
                loader=self.incremental_loader_name,
            )
            signal_mask = signal_mask_exclude_noise(y, self.noise_label)
            if signal_mask.any():
                preds = torch.argmax(masked_logits[signal_mask], dim=1)
                cls_tr_rec.append(macro_recall(preds, y.long()[signal_mask]))
            else:
                cls_tr_rec.append(0.0)
            metric_logits = masked_logits.detach()

            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            self.opt_wt.step()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return loss, avg_cls_tr_rec, metric_logits

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

        logits = misc_utils.apply_task_incremental_logit_mask(
            self.net.forward(x, fast_weights)[:, : self.n_outputs],
            t,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=t,
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )
        y_cls = unpack_y_to_class_labels(y).long()
        targets = y_cls
        loss = classification_cross_entropy(
            logits,
            targets,
            class_weighted_ce=self.class_weighted_ce,
        )

        if fast_weights is None:
            # fast_weights = self.net.parameters()
            fast_weights = list(self.net.parameters())

        graph_required = self.cfg.second_order

        # Some model parameters are intentionally frozen (e.g. adapter biases with
        # `requires_grad=False`). `torch.autograd.grad` errors if any of the
        # differentiation targets do not require grad, so we differentiate only
        # w.r.t. tensors that are set to require gradients and then reconstruct the
        # full gradient list aligned with `fast_weights`.
        require_grad_targets = [w for w in fast_weights if w.requires_grad]
        if require_grad_targets:
            raw_gradients_subset = torch.autograd.grad(
                loss,
                require_grad_targets,
                create_graph=graph_required,
                retain_graph=graph_required,
                allow_unused=True,
            )
            subset_iter = iter(raw_gradients_subset)
            raw_gradients = [
                next(subset_iter) if w.requires_grad else None for w in fast_weights
            ]
        else:
            raw_gradients = [None for _ in fast_weights]

        grads = [
            grad if grad is not None else torch.zeros_like(weight)
            for grad, weight in zip(raw_gradients, fast_weights)
        ]

        for i in range(len(grads)):
            if self.cfg.grad_clip_norm:
                clip_val = self.cfg.grad_clip_norm
                grads[i] = torch.clamp(grads[i], min=-clip_val, max=clip_val)

        updated_fast_weights = []
        for grad, weight, alpha_lr in zip(grads, fast_weights, self.net.alpha_lr):
            # Preserve frozen tensors exactly; only update tensors that are
            # intended to participate in gradient-based inner updates.
            if not weight.requires_grad:
                updated_fast_weights.append(weight)
                continue
            updated_fast_weights.append(weight - grad * alpha_lr)
        fast_weights = updated_fast_weights
        return fast_weights, loss.item()

    def la_ER(self, raw_x, y, t):
        """
        this ablation tests whether it suffices to just do the learning rate modulation
        guided by gradient alignment + clipping (that La-MAML does implciitly through autodiff)
        and use it with ER (therefore no meta-learning for the weights)

        """
        cls_tr_rec = []
        # Class labels aligned with ``raw_x`` (before any per-pass shuffle), used
        # for the live current-batch weight-update loss below.
        current_labels = unpack_y_to_class_labels(y).long()
        for pass_itr in range(self.inner_steps):
            # Rebuild a fresh canonicalized tensor each round; previous
            # autograd.grad/backward calls free the old graph.
            x = self._canonicalize_input(raw_x, detach=False)

            perm = torch.randperm(x.size(0))
            x = x[perm]
            if isinstance(y, (list, tuple)):
                y = tuple(yi[perm] if yi is not None else None for yi in y)
            else:
                y = y[perm]

            batch_sz = x.shape[0]
            n_batches = self.cfg.meta_batches
            rough_sz = math.ceil(batch_sz / n_batches)
            fast_weights = None
            meta_losses = [0 for _ in range(n_batches)]

            y_pack = unpack_y_to_class_labels(y)
            bx, by, bt, replay_count = self.getBatch(
                x.detach().cpu().numpy(),
                y_pack.detach().cpu().numpy(),
                t,
            )
            bx = bx.squeeze()

            for i in range(n_batches):

                batch_x = x[i * rough_sz : (i + 1) * rough_sz]
                if isinstance(y, (list, tuple)):
                    batch_y = tuple(
                        (
                            yi[i * rough_sz : (i + 1) * rough_sz]
                            if yi is not None
                            else None
                        )
                        for yi in y
                    )
                else:
                    batch_y = y[i * rough_sz : (i + 1) * rough_sz]

                # assuming labels for inner update are from the same
                fast_weights, inner_loss = self.inner_update(
                    batch_x, fast_weights, batch_y, t
                )

                prediction = self.net.forward(bx, fast_weights)
                meta_loss = self._weighted_multitask_loss(
                    prediction, by, bt, replay_count
                )
                meta_losses[i] += meta_loss

            # update alphas
            self.net.zero_grad()
            self.opt_lr.zero_grad()

            meta_loss = meta_losses[-1]  # sum(meta_losses)/len(meta_losses)
            meta_loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.net.alpha_lr.parameters(), self.cfg.grad_clip_norm
                )

            # update the LRs (guided by meta-loss, but not the weights)
            self.opt_lr.step()

            # update weights
            self.net.zero_grad()

            # Compute the ER loss for the network weights. The current rows flow
            # through a freshly canonicalized, adapter-differentiable forward on
            # ``raw_x`` (the per-pass ``x`` graph was already consumed by the
            # meta backward above), so the input adapter receives gradients in
            # the same backward as the backbone. Replay rows reuse the
            # pre-canonicalized buffer tensors.
            x_live = self._canonicalize_input(raw_x, detach=False)
            current_t = torch.full(
                (x_live.size(0),), int(t), dtype=torch.long, device=x_live.device
            )
            current_logits = self.net.forward(x_live)
            current_loss = self.take_multitask_loss(
                current_t, current_logits, current_labels
            )
            if replay_count > 0:
                replay_logits = self.net.forward(bx[:replay_count])
                replay_loss = self.take_multitask_loss(
                    bt[:replay_count], replay_logits, by[:replay_count]
                )
            else:
                replay_loss = torch.zeros(
                    (), device=current_logits.device, dtype=current_logits.dtype
                )
            loss = current_loss + (self.memory_loss_lambda * replay_loss)
            cls_tr_rec.append(
                self._batch_accuracy(current_t, current_logits, current_labels)
            )

            loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            # update weights with grad from simple ER loss
            # and LRs obtained from meta-loss guided by old and new tasks
            for i, p in enumerate(self.net.parameters()):
                if p.grad is None:
                    continue
                p.data = p.data - (p.grad * nn.functional.relu(self.net.alpha_lr[i]))
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        # The meta path computes its metric on the combined replay+current batch
        # via ``_batch_accuracy``; no current-batch masked logits are exposed for
        # the progress bar, so ``main.py`` falls back to a separate eval forward.
        return loss, avg_cls_tr_rec, None
