"""Uncertainty-guided Continual Learning (UCL) implementation for ``main.py``.

This rewrite adapts the original reference code – which relied on an external
training loop – so that it exposes the standard ``Net`` interface used across
this repository.  The core ingredients of UCL remain intact: a Bayesian linear
head with learnable mean and log-variance parameters, the KL-style penalties on
``mu``/``sigma`` drift, and the per-task snapshot that anchors the posterior of
all subsequent tasks.

For practicality we reuse the existing ``ResNet1D`` backbone as a deterministic
feature extractor while keeping the Bayesian machinery on the classifier
weights.  This mirrors the original setting where low-level filters are
deterministic and only the task-specific classifier carries uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall
from utils import misc_utils


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Tensor needs at least 2 dims to compute fan in/out")
    if tensor.dim() == 2:  # Linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field
        fan_out = num_output_fmaps * receptive_field
    return fan_in, fan_out


class Gaussian:
    """Reparameterised Gaussian for Bayesian layers."""

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor) -> None:
        self.mu = mu
        self.rho = rho
        self._normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.rho))

    def sample(self) -> torch.Tensor:
        eps = self._normal.sample(self.mu.size()).to(self.mu.device)
        return self.mu + self.sigma * eps


class BayesianLinear(nn.Module):
    """Factorised Gaussian linear layer mirroring the UCL implementation."""

    def __init__(self, in_features: int, out_features: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio # spread of posterior over mu of weights (epistemic uncertainty)
        mu_var = total_var - noise_var # init variance of mu (to enable learning)

        noise_std = noise_var**0.5
        mu_std = mu_var**0.5
        bound = (3.0**0.5) * mu_std 
        nn.init.uniform_(self.weight_mu, -bound, bound) # init uniform distr. for mu in [-bound, bound]

        rho_init = float(torch.log(torch.expm1(torch.tensor(noise_std)))) # std = log(1 + exp(rho)) => rho = log(exp(std) - 1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init)) # each weight has its own rho

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.linear(x, weight, self.bias)


@dataclass
class UCLConfig:
    lr: float = 1e-3
    lr_rho: float = 1e-2
    beta: float = 0.0002
    alpha: float = 0.3
    ratio: float = 0.125
    clipgrad: float = 10.0
    split: bool = True

    @staticmethod
    def from_args(args: object) -> "UCLConfig":
        cfg = UCLConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class BayesianClassifier(nn.Module):
    """ResNet1D feature extractor followed by per-task Bayesian heads."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int,
                 cfg: UCLConfig, args: object | None, classes_per_task: Optional[List[int]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.n_outputs = n_outputs
        if classes_per_task is None:
            classes_per_task = misc_utils.build_task_class_list(
                n_tasks,
                n_outputs,
                nc_per_task=None,
                classes_per_task=None,
            )

        # Feature extractor (deterministic)
        self.feature_net = ResNet1D(n_outputs, args)
        self.feature_dim = self.feature_net.model.fc.in_features

        # Replace the original classifier; heads operate on extracted features.
        self.feature_net.model.fc = nn.Identity()

        self.heads = nn.ModuleList(
            [BayesianLinear(self.feature_dim, c, ratio=cfg.ratio)
             for c in classes_per_task]
        )

        self.split = cfg.split

    def forward(self, x: torch.Tensor, sample: bool = False) -> List[torch.Tensor] | torch.Tensor:
        feats = self.feature_net.forward(x, ret_feats=True)
        if isinstance(self.heads, nn.ModuleList):
            outputs = [head(feats, sample=sample) for head in self.heads]
        else:
            outputs = self.heads(feats, sample=sample)
        if self.split:
            return outputs
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    """UCL learner compatible with ``main.py``."""

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()

        self.cfg = UCLConfig.from_args(args)
        assert n_tasks > 0, "Number of tasks must be positive for UCL"

        self.args = args
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)

        self.model = BayesianClassifier(n_inputs, n_outputs, n_tasks, self.cfg, args, self.classes_per_task)
        self.split = self.cfg.split

        # Optimiser: deterministic layers + Bayesian mu share lr, rho parameters get lr_rho
        mu_params: List[nn.Parameter] = []
        rho_params: List[nn.Parameter] = []
        for head in self.model.heads:
            mu_params.extend([head.weight_mu, head.bias])
            rho_params.append(head.weight_rho)
        # mu_params.extend([self.model.heads.weight_mu, self.model.heads.bias])
        # rho_params.append(self.model.heads.weight_rho)
        mu_params.extend(p for p in self.model.feature_net.parameters())

        self.optimizer = torch.optim.Adam(
            [
                {"params": mu_params, "lr": self.cfg.lr},
                {"params": rho_params, "lr": self.cfg.lr_rho},
            ],
            lr=self.cfg.lr,
        )

        self.current_task: Optional[int] = None
        self.model_old: Optional[BayesianClassifier] = None
        self.saved = False
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.is_task_incremental: bool = True

    # ------------------------------------------------------------------
    def compute_offsets(self, task):
            if self.is_task_incremental:
                offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
            else:
                offset1 = 0
                offset2 = self.n_outputs
            return int(offset1), int(offset2)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, t: int, s: Optional[float] = None) -> torch.Tensor:
        outputs = self.model(x, sample=False)
        if self.split:
            # offset1, offset2 = self.compute_offsets(t)
            return outputs[t]
        return outputs

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if (self.current_task is None) or (t != self.current_task):
            if self.current_task is not None:
                self.model_old = self._snapshot_model()
                self.saved = True
            self.current_task = t

        device = self._device()

        if self.split:
            offset1, offset2 = self.compute_offsets(t)
            y_local = y.clone()
            y_local = y_local - offset1
            task_classes = self.classes_per_task[t]
            if (y_local.min() < 0 ) or (y_local.max() >= task_classes):
                raise ValueError(f"Labels out of range for task {t}: expected in [0, {task_classes-1}] after offset, got [{int(y_local.min())}, {int(y_local.max())}]")
            y = y_local

        x = x.to(device)
        y = y.to(device)

        self.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = False
        
        # offset1, offset2 = self.compute_offsets(t)
        outputs = self.model(x, sample=True)
        logits = outputs[t] if self.split else outputs

        preds = torch.argmax(logits, dim=1)
        tr_acc = macro_recall(preds, y)
        loss = self.ce(logits, y)
        loss = self._apply_regularisation(loss, y.size(0))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clipgrad)
        self.optimizer.step()

        return float(loss.detach().cpu()), tr_acc

    def on_epoch_end(self) -> None:  # pragma: no cover - hook for symmetry
        pass

    # ------------------------------------------------------------------
    def _snapshot_model(self) -> BayesianClassifier:
        clone: BayesianClassifier = BayesianClassifier(
            self.n_inputs,
            self.n_outputs,
            self.n_tasks,
            self.cfg,
            self.args,
        )
        clone.load_state_dict(self.model.state_dict())
        clone.to(self._device())
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone

    def _apply_regularisation(self, base_loss: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not self.saved or self.model_old is None:
            return base_loss

        sigma_weight_reg = torch.zeros_like(base_loss)
        sigma_weight_normal_reg = torch.zeros_like(base_loss)
        mu_weight_reg = torch.zeros_like(base_loss)
        mu_bias_reg = torch.zeros_like(base_loss)
        l1_mu_weight_reg = torch.zeros_like(base_loss)
        l1_mu_bias_reg = torch.zeros_like(base_loss)

        eps = 1e-8
        for old_head, new_head in zip(self.model_old.heads, self.model.heads):
            trainer_weight_mu = new_head.weight_mu # t = t
            saver_weight_mu = old_head.weight_mu # t = t-1
            trainer_bias = new_head.bias
            saver_bias = old_head.bias

            trainer_weight_sigma = torch.log1p(torch.exp(new_head.weight_rho)) # noise var
            saver_weight_sigma = torch.log1p(torch.exp(old_head.weight_rho)) # noise var

            fan_in, _ = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            std_init = math.sqrt((2.0 / fan_in) * self.cfg.ratio)

            saver_strength = std_init / (saver_weight_sigma + eps)
            bias_strength = saver_strength.mean(dim=1)

            mu_weight_reg = mu_weight_reg + ((saver_strength * (trainer_weight_mu - saver_weight_mu)) ** 2).sum()
            mu_bias_reg = mu_bias_reg + ((bias_strength * (trainer_bias - saver_bias)) ** 2).sum()

            l1_mu_weight_reg = l1_mu_weight_reg + (
                (saver_weight_mu.pow(2) / saver_weight_sigma.pow(2))
                * (trainer_weight_mu - saver_weight_mu).abs()
            ).sum()
            l1_mu_bias_reg = l1_mu_bias_reg + (
                (saver_bias.pow(2) / (saver_weight_sigma.mean(dim=1).pow(2) + eps))
                * (trainer_bias - saver_bias).abs()
            ).sum()

            weight_sigma_ratio = trainer_weight_sigma.pow(2) / (saver_weight_sigma.pow(2) + eps)
            sigma_weight_reg = sigma_weight_reg + (weight_sigma_ratio - torch.log(weight_sigma_ratio + eps)).sum()
            sigma_weight_normal_reg = sigma_weight_normal_reg + (
                trainer_weight_sigma.pow(2) - torch.log(trainer_weight_sigma.pow(2) + eps)
            ).sum()

        loss = base_loss
        loss = loss + self.cfg.alpha * (mu_weight_reg + mu_bias_reg) / (2 * batch_size)
        loss = loss + self.saved * (l1_mu_weight_reg + l1_mu_bias_reg) / batch_size
        loss = loss + self.cfg.beta * (sigma_weight_reg + sigma_weight_normal_reg) / (2 * batch_size)
        return loss
    
    @torch.no_grad()
    def mc_epistemic_classification(self, x, t, S=20, temperature=1.0, clamp_eps=1e-8):
        """
        Monte-Carlo epistemic uncertainty for classification.
        Assumes model(x, sample=True/False) returns logits.
        Returns:
            p_mean: (B, C) predictive probabilities
            H_pred: (B,) predictive entropy H[p_mean]
            EH:     (B,) expected entropy E_s[ H[p_s] ]
            MI:     (B,) mutual information H[p_mean] - E_s[H[p_s]]  (epistemic)
        """
        model = self.model
        model.eval()
        logits_accum = []
        probs_accum  = []

        for _ in range(S):
            logits = model(x, sample=True)[t] #/ temperature                  # (B, C)
            probs  = F.softmax(logits, dim=-1)                            # (B, C)
            # logits_accum.append(logits)
            probs_accum.append(probs)

        # Stack over samples
        probs_stack = torch.stack(probs_accum, dim=0)                     # (S, B, C)

        # Predictive mean probability
        p_mean = probs_stack.mean(dim=0)                                  # (B, C)

        # Entropy of the mean H[p_mean]
        p_mean_clamped = p_mean.clamp(min=clamp_eps, max=1.0)
        H_pred = -(p_mean_clamped * p_mean_clamped.log()).sum(dim=-1)     # (B,)

        # Expected entropy E_s[ H[p_s] ]
        probs_clamped = probs_stack.clamp(min=clamp_eps, max=1.0)
        entropies = -(probs_clamped * probs_clamped.log()).sum(dim=-1)    # (S, B)
        EH = entropies.mean(dim=0)                                        # (B,)

        # Mutual information (epistemic)
        MI = H_pred - EH                                                   # (B,)

        return p_mean, H_pred, EH, MI



__all__ = ["Net"]
