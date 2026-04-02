# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as F

import numpy as np
import random

import sys
from model.resnet1d import ResNet1D
from model.detection_replay import (
    DetectionReplayMixin,
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataloaders"))
)
from iq_data_loader import ensure_iq_two_channel
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("once")


@dataclass
class IcarlConfig:
    lr: float = 1e-3
    memory_strength: float = 0.5
    n_memories: int = 5120
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
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = IcarlConfig.from_args(args)
        self.nt = n_tasks
        self.reg = self.cfg.memory_strength
        self.n_memories = self.cfg.n_memories
        self.num_exemplars = 0
        # Classification operates in logit space (n_classes) but iCaRL's
        # nearest-mean classifier should use penultimate-layer features.
        self.n_classes = n_outputs
        # Initialise n_feat conservatively; will be overwritten once the
        # backbone is constructed and exposes its feature dimension.
        self.n_feat = n_outputs
        self.samples_per_task_resolver = getattr(args, "get_samples_per_task", None)
        self.samples_per_task = (
            self.cfg.samples_per_task
        )  # * (1.0 - self.cfg.validation)
        if self.samples_per_task_resolver is None:
            assert self.samples_per_task > 0, "Samples per task is <= 0"
        self.examples_seen = 0

        self.glances = self.cfg.glances
        # setup network

        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)
        # Use the backbone's penultimate feature size for iCaRL embeddings.
        self.n_feat = getattr(self.net, "feature_dim", self.n_classes)

        # setup optimizer
        self.opt = torch.optim.SGD(self._ll_params(), lr=self.cfg.lr, momentum=0.9)
        self.det_opt = torch.optim.SGD(
            self.net.det_head.parameters(), lr=self.cfg.lr, momentum=0.9
        )

        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        # Use batchmean to follow KL definition and avoid PyTorch warning
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        print(self.n_memories, self.reg, self.det_lambda, self.samples_per_task)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        # memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}

        self.gpu = self.cfg.cuda
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.n_outputs = n_outputs
        self.noise_label: int | None = noise_label_from_args(args)

    def _ensure_iq_shape(self, x):
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            L = F // 2
            return x.view(B, 2, L)
        raise ValueError(
            f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L)."
        )

    def _prepare_input(self, x):
        if self.cfg.dataset == "tinyimagenet":
            return x.view(-1, 3, 64, 64)
        if self.cfg.dataset == "cifar100":
            return x.view(-1, 3, 32, 32)
        if self.is_iq:
            return self._ensure_iq_shape(x)
        return x

    def _prepare_det_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._prepare_input(x)

    def netforward(self, x):
        if self.cfg.dataset == "tinyimagenet":
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == "cifar100":
            x = x.view(-1, 3, 32, 32)
        elif "iq" in self.cfg.dataset.lower():
            x_np = ensure_iq_two_channel(x.detach().cpu().numpy())
            x = torch.from_numpy(x_np).float()
            if self.gpu:
                x = x.cuda()

        return self.net.forward(x)

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns penultimate-layer features.

        Mirrors ``netforward`` input handling so that exemplar construction and
        nearest-mean classification operate in feature space rather than over
        logits.
        """
        if self.cfg.dataset == "tinyimagenet":
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == "cifar100":
            x = x.view(-1, 3, 32, 32)
        elif "iq" in self.cfg.dataset.lower():
            x_np = ensure_iq_two_channel(x.detach().cpu().numpy())
            x = torch.from_numpy(x_np).float()
            if self.gpu:
                x = x.cuda()

        return self.net.forward_features(x)

    def compute_offsets(self, task):
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def _get_samples_per_task(self, task):
        if self.samples_per_task_resolver is None:
            return self.samples_per_task
        return int(self.samples_per_task_resolver(task))

    def forward(
        self,
        x: torch.Tensor,
        t: int,
        *,
        cil_all_seen_upto_task: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Classify with the iCaRL nearest-class-mean (NCM) rule over exemplars.

        Per Rebuffi et al., CVPR 2017, inference uses **all** stored class
        prototypes (every class with exemplars), not only the current task's
        label block. This makes the built-in evaluator compatible with
        class-incremental protocols. The ``cil_all_seen_upto_task`` keyword is
        accepted for API parity with other learners and is ignored here because
        the exemplar set already defines the active class set.

        Args:
            x: Input batch.
            t: Current task index (used only when no exemplars exist yet).
            cil_all_seen_upto_task: Ignored (NCM uses all classes with exemplars).
            **kwargs: Swallows extra keys from the training loop.

        Returns:
            One-hot style predictions ``(batch, n_classes)`` (large negative
            mass off the predicted class), matching the previous interface.
        """
        del kwargs
        _ = cil_all_seen_upto_task

        ns = x.size(0)
        device = x.device
        dtype = torch.float32
        task_classes = self.classes_per_task[t]
        offset1, offset2 = self.compute_offsets(t)

        class_ids = sorted(self.mem_class_x.keys())
        if not class_ids:
            out = torch.full(
                (ns, self.n_classes),
                -1e10,
                device=device,
                dtype=dtype,
            )
            block = max(task_classes, 1)
            out[:, offset1:offset2] = 1.0 / block
            return out

        means_rows: list[torch.Tensor] = []
        for class_id in class_ids:
            exemplars = self.mem_class_x[class_id]
            means_rows.append(self.feature_forward(exemplars).detach().mean(dim=0))
        means = torch.stack(means_rows, dim=0)
        feats = self.feature_forward(x).detach()
        means = F.normalize(means, p=2, dim=1)
        feats = F.normalize(feats, p=2, dim=1)
        distances = torch.cdist(feats, means, p=2)
        nearest = distances.argmin(dim=1)
        id_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)
        pred_labels = id_tensor[nearest]

        out = torch.full((ns, self.n_classes), -1e10, device=device, dtype=dtype)
        out.scatter_(1, pred_labels.unsqueeze(1), 1.0)
        return out

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
            output[:, offset2 : self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, y, t):

        batch_count = x.size(0)
        y_cls = unpack_y_to_class_labels(y)
        self.net.train()
        if self.gpu:
            self.net.cuda()

        cls_tr_rec = []

        for pass_itr in range(self.glances):

            # only make changes like pushing to buffer once per batch and not for every glance
            if pass_itr == 0:
                # Track task progress in terms of full training batches so the
                # end-of-task trigger remains correct for n_epochs > 1.
                prev_examples_seen = self.examples_seen
                self.examples_seen += batch_count
                samples_per_task = self._get_samples_per_task(t)
                assert samples_per_task > 0, "Samples per task is <= 0"

                # Stage only the first pass through the task data. This keeps
                # exemplar construction bounded while still supporting n_epochs > 1.
                if prev_examples_seen < samples_per_task:
                    if self.memx is None:
                        self.memx = x.data.clone()
                        self.memy = y_cls.data.clone()
                    else:
                        self.memx = torch.cat((self.memx, x.data.clone()))
                        self.memy = torch.cat((self.memy, y_cls.data.clone()))

            self.net.zero_grad()
            offset1, _offset2 = self.compute_offsets(t)
            logits_full = misc_utils.apply_task_incremental_logit_mask(
                self.netforward(x),
                t,
                self.classes_per_task,
                self.n_classes,
                cil_all_seen_upto_task=t,
                global_noise_label=self.noise_label,
            )
            signal_mask = signal_mask_exclude_noise(y_cls, self.noise_label)
            targets = y_cls.long()
            if signal_mask.any():
                preds = torch.argmax(logits_full[signal_mask], dim=1)
                cls_tr_rec.append(macro_recall(preds, targets[signal_mask]))
            else:
                cls_tr_rec.append(0.0)
            loss = classification_cross_entropy(
                logits_full,
                targets,
                class_weighted_ce=self.class_weighted_ce,
            )

            # num_exemplars remains 0 unless final epoch is reached
            if self.num_exemplars > 0:
                # distillation
                for tt in range(t):
                    # first generate a minibatch with one example per class from
                    # previous tasks
                    task_classes = self.classes_per_task[tt]
                    input_shape = x.shape[1:]
                    inp_dist = x.new_zeros((task_classes,) + input_shape)
                    # Distillation operates over classifier logits, which have
                    # dimension ``n_classes`` rather than the feature size.
                    target_dist = x.new_zeros((task_classes, self.n_classes))
                    offset1, offset2 = self.compute_offsets(tt)
                    for cc in range(task_classes):
                        indx = random.randint(
                            0, len(self.mem_class_x[cc + offset1]) - 1
                        )
                        inp_dist[cc] = self.mem_class_x[cc + offset1][indx].clone()
                        target_dist[cc] = self.mem_class_y[cc + offset1][indx].clone()
                    # Add distillation loss
                    loss += (
                        self.reg
                        * self.kl(
                            self.lsm(self.netforward(inp_dist)[:, offset1:offset2]),
                            self.sm(target_dist[:, offset1:offset2]),
                        )
                        * task_classes
                    )
            # bprop and update
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.cfg.grad_clip_norm
                )

            self.opt.step()

        # Check whether this is the last minibatch of the current task across
        # all configured epochs.
        target = int(self.cfg.n_epochs * self._get_samples_per_task(t))
        # print(f"Samples per task: {self._get_samples_per_task(t)}, n_epochs: {self.cfg.n_epochs}")
        # print(f"Target samples for task {t}: {target}, examples seen: {self.examples_seen}")
        if self.examples_seen >= target:  # not ==
            # print(f"Final batch for task {t} reached. Updating exemplar memory.")
            self.examples_seen = 0
            # self._rebuild_exemplars_for_task(t, x.device)

            # if self.examples_seen == self.cfg.n_epochs * self.samples_per_task:
            #     self.examples_seen = 0
            # get labels from previous task; we assume labels are consecutive
            if self.memx is None or self.memy is None or self.memy.numel() == 0:
                return float(loss.item()), (
                    sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
                )

            offset1, offset2 = self.compute_offsets(t)
            if self.gpu:
                all_labs = torch.LongTensor(np.unique(self.memy.cpu().numpy()))
            else:
                all_labs = torch.LongTensor(np.unique(self.memy.numpy()))

            # Per-task signal slice plus global noise (same id across IQ tasks) when present.
            in_task = (all_labs >= offset1) & (all_labs < offset2)
            signal_labs = all_labs[in_task]
            noise_key = self.noise_label
            has_noise_exemplars = (
                noise_key is not None
                and 0 <= int(noise_key) < self.n_classes
                and (all_labs == int(noise_key)).any()
            )
            if has_noise_exemplars:
                noise_tensor = signal_labs.new_tensor(
                    [int(noise_key)], dtype=signal_labs.dtype
                )
                task_labs = torch.cat([signal_labs, noise_tensor])
            else:
                task_labs = signal_labs
            task_labs, _ = torch.sort(task_labs)
            num_classes = task_labs.size(0)

            # print("num_classes", num_classes, "nc_per_task", self.nc_per_task, "offsets",
            #       offset1, offset2)
            current_task_classes = self.classes_per_task[t]
            if signal_labs.size(0) != current_task_classes:
                print(
                    "[WARNING][iCaRL] Task {} expected {} classes, found {} in memory.".format(
                        t, current_task_classes, signal_labs.size(0)
                    )
                )
            if num_classes > 0:
                # Reduce exemplar set by updating value of num. exemplars per class
                self.num_exemplars = int(
                    self.n_memories / (num_classes + len(self.mem_class_x.keys()))
                )
                for ll in range(num_classes):
                    label = task_labs[ll]  # current label
                    indxs = (
                        (self.memy == label).nonzero(as_tuple=False).view(-1)
                    )  # indices of current label
                    cdata = self.memx.index_select(
                        0, indxs
                    )  # grab training data for current label
                    # Construct exemplar set for last task using penultimate
                    # features as in the original iCaRL algorithm.
                    feat_cdata = self.feature_forward(cdata).data.clone()
                    mean_feature = feat_cdata.mean(0)
                    nd = self.n_feat
                    exemplars = cdata.new_zeros((self.num_exemplars,) + cdata.shape[1:])
                    ntr = cdata.size(0)  # num data points for current label
                    # used to keep track of which examples we have already used
                    taken = torch.zeros(ntr)
                    model_output = feat_cdata
                    for ee in range(self.num_exemplars):  # herding loop
                        prev = torch.zeros(1, nd)
                        if self.gpu:
                            prev = prev.cuda()
                        if ee > 0:
                            prev = (
                                self.feature_forward(exemplars[:ee]).data.clone().sum(0)
                            )
                        cost = (
                            (
                                mean_feature.expand(ntr, nd)
                                - (model_output + prev.expand(ntr, nd)) / (ee + 1)
                            )
                            .norm(2, 1)
                            .squeeze()
                        )
                        _, indx = cost.sort(0)  # sort by ascending cost
                        winner = 0
                        while winner < indx.size(0) and taken[indx[winner]] == 1:
                            winner += 1
                        if winner < indx.size(0):
                            taken[indx[winner]] = 1
                            exemplars[ee] = cdata[indx[winner]].clone()
                        else:
                            exemplars = exemplars[: indx.size(0)].clone()
                            self.num_exemplars = indx.size(0)
                            break
                    # update memory with exemplars
                    self.mem_class_x[label.item()] = exemplars.clone()

                # recompute outputs for distillation purposes
                for cc in self.mem_class_x.keys():
                    self.mem_class_x[cc] = self.mem_class_x[cc][: self.num_exemplars]
                    self.mem_class_y[cc] = self.netforward(
                        self.mem_class_x[cc]
                    ).data.clone()
            self.memx = None
            self.memy = None
            # print(len(self.mem_class_x[0]))

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        det_loss_value = 0.0
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
        #     det_loss_value = float(det_loss.item())
        total_loss = float(loss.item()) + det_loss_value
        # det_pred = (det_logits >= 0).long()
        # det_recall = macro_recall(det_pred, y_det.long())
        # neg_mask = y_det == 0
        # if neg_mask.any():
        #     neg_preds = det_pred[neg_mask]
        #     fp = (neg_preds == 1).sum().item()
        #     tn = (neg_preds == 0).sum().item()
        #     denom = fp + tn
        #     det_pfa = float(fp / denom) if denom > 0 else 0.0
        # else:
        #     det_pfa = 0.0
        # score = avg_cls_tr_rec * det_recall * (1.0 - det_pfa)
        # print(
        #     f"Task {t} | Score: {score:.4f} | Loss: {total_loss:.4f} | Cls Loss: {loss.item():.4f} "
        #     f"| Det Loss: {det_loss.item():.4f} | Det Recall: {det_recall:.4f} | Det PFA: {det_pfa:.4f} "
        #     f"| Det_lambda: {self.det_lambda} | Memory Strength: {self.reg}"
        # )
        return total_loss, avg_cls_tr_rec
