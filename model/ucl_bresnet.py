"""Uncertainty-guided Continual Learning with a Bayesian 1D ResNet backbone.

This variant mirrors the behaviour of :mod:`model.ucl` but replaces the
deterministic ``ResNet1D`` feature extractor with a fully Bayesian
counterpart.  All convolutional layers now maintain Gaussian posteriors over
their weights, enabling epistemic uncertainty estimation deeper in the
network while keeping the multi-head Bayesian linear classifiers used for
task-specific outputs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BayesianLayer(nn.Module):
    """Mixin-style base class exposing Bayesian parameters."""

    weight_mu: nn.Parameter
    weight_rho: nn.Parameter

    @property
    def weight_sigma(self) -> torch.Tensor:
        return torch.log1p(torch.exp(self.weight_rho))

    def mu_parameters(self) -> Iterable[nn.Parameter]:
        for attr in ("weight_mu", "bias"):
            param = getattr(self, attr, None)
            if isinstance(param, nn.Parameter):
                yield param

    def rho_parameters(self) -> Iterable[nn.Parameter]:
        yield self.weight_rho


class BayesianLinear(BayesianLayer):
    """Factorised Gaussian linear layer mirroring the UCL implementation."""

    def __init__(self, in_features: int, out_features: int, ratio: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std = noise_var ** 0.5
        mu_std = mu_var ** 0.5
        bound = (3.0 ** 0.5) * mu_std
        nn.init.uniform_(self.weight_mu, -bound, bound)

        rho_init = float(math.log(math.expm1(noise_std)))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.linear(x, weight, self.bias)


class BayesianConv1d(BayesianLayer):
    """1D convolution with Gaussian weight posterior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("BayesianConv1d currently models bias-free convolutions")

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weight_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight_mu = nn.Parameter(torch.Tensor(*weight_shape))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        total_var = 2.0 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var

        noise_std = noise_var ** 0.5
        mu_std = mu_var ** 0.5
        bound = (3.0 ** 0.5) * mu_std
        nn.init.uniform_(self.weight_mu, -bound, bound)

        rho_init = float(math.log(math.expm1(noise_std)))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, rho_init))

        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        weight = self.weight.sample() if sample else self.weight.mu
        return F.conv1d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BayesianConvBN1d(nn.Module):
    """Convenience wrapper for Conv1d -> BatchNorm1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        ratio: float,
    ) -> None:
        super().__init__()
        self.conv = BayesianConv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            ratio=ratio,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        x = self.conv(x, sample=sample)
        x = self.bn(x)
        return x


class BayesianBasicBlock1D(nn.Module):
    """1D version of ResNet BasicBlock with Bayesian convolutions."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        ratio: float,
    ) -> None:
        super().__init__()
        self.conv1 = BayesianConv1d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            ratio=ratio,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BayesianConv1d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            ratio=ratio,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        identity = x

        out = self.conv1(x, sample=sample)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, sample=sample)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, sample=sample)

        out += identity
        out = self.relu(out)
        return out


class BayesianResNet1D(nn.Module):
    """ResNet-18 style network with Bayesian convolutions for IQ data."""

    def __init__(self, in_channels: int = 2, ratio: float = 0.5) -> None:
        super().__init__()
        self.inplanes = 64
        self.ratio = ratio

        self.conv1 = BayesianConv1d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=1,
            ratio=ratio,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512 * BayesianBasicBlock1D.expansion

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.ModuleList:
        downsample = None
        if stride != 1 or self.inplanes != planes * BayesianBasicBlock1D.expansion:
            downsample = BayesianConvBN1d(
                self.inplanes,
                planes * BayesianBasicBlock1D.expansion,
                stride=stride,
                ratio=self.ratio,
            )

        layers = nn.ModuleList()
        layers.append(
            BayesianBasicBlock1D(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                ratio=self.ratio,
            )
        )
        self.inplanes = planes * BayesianBasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(
                BayesianBasicBlock1D(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    ratio=self.ratio,
                )
            )
        return layers

    def _forward_layer(self, layer: nn.ModuleList, x: torch.Tensor, sample: bool) -> torch.Tensor:
        for block in layer:
            x = block(x, sample=sample)
        return x

    def forward(self, x: torch.Tensor, sample: bool = False, ret_feats: bool = False) -> torch.Tensor:
        x = self.conv1(x, sample=sample)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self._forward_layer(self.layer1, x, sample)
        x = self._forward_layer(self.layer2, x, sample)
        x = self._forward_layer(self.layer3, x, sample)
        x = self._forward_layer(self.layer4, x, sample)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if ret_feats:
            return x
        return x  # feature extractor only; heads perform classification


@dataclass
class UCLConfig:
    lr: float = 1e-3
    lr_rho: float = 1e-2
    beta: float = 0.0002
    alpha: float = 0.3
    ratio: float = 0.125
    clipgrad: float = 10.0
    split: bool = True
    eval_samples: int = 20

    @staticmethod
    def from_args(args: object) -> "UCLConfig":
        cfg = UCLConfig()
        # if hasattr(args, "clipgrad"):
        #     cfg.clipgrad = getattr(args, "clipgrad")
        # if hasattr(args, "split"):
        #     cfg.split = getattr(args, "split")
        # if hasattr(args, "ratio"):
        #     cfg.ratio = getattr(args, "ratio")
        # if hasattr(args, "eval_samples"):
        #     cfg.eval_samples = max(1, int(getattr(args, "eval_samples")))
        return cfg


class BayesianClassifier(nn.Module):
    """Bayesian ResNet feature extractor followed by per-task Bayesian heads."""

    def __init__(self, n_outputs: int, n_tasks: int, cfg: UCLConfig, args: object | None, classes_per_task=None) -> None:
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

        self.feature_net = BayesianResNet1D(in_channels=getattr(args, "in_channels", 2), ratio=cfg.ratio)
        self.feature_dim = self.feature_net.feature_dim

        self.heads = nn.ModuleList(
            [BayesianLinear(self.feature_dim, c, ratio=cfg.ratio) for c in classes_per_task]
        )

        self.split = cfg.split

    def forward(self, x: torch.Tensor, sample: bool = False) -> List[torch.Tensor] | torch.Tensor:
        feats = self.feature_net(x, sample=sample, ret_feats=True)
        outputs = [head(feats, sample=sample) for head in self.heads]
        if self.split:
            return outputs
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    """UCL learner powered by a Bayesian ResNet-18 backbone."""

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

        self.model = BayesianClassifier(n_outputs, n_tasks, self.cfg, args, self.classes_per_task)
        self.split = self.cfg.split

        mu_params: List[nn.Parameter] = []
        rho_params: List[nn.Parameter] = []

        for module in self._iter_bayesian_modules(self.model):
            mu_params.extend(module.mu_parameters())
            rho_params.extend(module.rho_parameters())

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

    @contextmanager
    def _temporarily_enable_bn_training(self):
        bn_modules: List[nn.BatchNorm1d] = []
        states: List[bool] = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_modules.append(module)
                states.append(module.training)
                module.train(True)
        try:
            yield
        finally:
            for module, state in zip(bn_modules, states):
                module.train(state)

    # ------------------------------------------------------------------
    def compute_offsets(self, task: int) -> Tuple[int, int]:
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, t: int, s: Optional[float] = None) -> torch.Tensor:
        if not self.training:
            num_samples = max(1, self.cfg.eval_samples)
            if num_samples == 1:
                outputs = self.model(x, sample=False)
                return outputs[t] if self.split else outputs

            probs_acc: Optional[torch.Tensor] = None
            with torch.no_grad():
                with self._temporarily_enable_bn_training():
                    for _ in range(num_samples):
                        sampled = self.model(x, sample=True)
                        head_logits = sampled[t] if self.split else sampled
                        head_probs = F.softmax(head_logits, dim=-1)
                        probs_acc = head_probs if probs_acc is None else probs_acc + head_probs

            assert probs_acc is not None
            probs_mean = probs_acc / float(num_samples)
            return torch.log(probs_mean.clamp_min(1e-8))

        outputs = self.model(x, sample=False)
        return outputs[t] if self.split else outputs

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> Tuple[float, float]:
        if (self.current_task is None) or (t != self.current_task):
            if self.current_task is not None:
                self.model_old = self._snapshot_model()
                self.saved = True
            self.current_task = t

        device = self._device()

        if self.split:
            offset1, _ = self.compute_offsets(t)
            y_local = y.clone() - offset1
            task_classes = self.classes_per_task[t]
            if (y_local.min() < 0) or (y_local.max() >= task_classes):
                raise ValueError(
                    f"Labels out of range for task {t}: expected in [0, {task_classes - 1}] after offset, got "
                    f"[{int(y_local.min())}, {int(y_local.max())}]"
                )
            y = y_local

        x = x.to(device)
        y = y.to(device)

        self.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = False

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

    def on_epoch_end(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    def _snapshot_model(self) -> BayesianClassifier:
        clone = BayesianClassifier(self.n_outputs, self.n_tasks, self.cfg, self.args)
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

        for old_layer, new_layer in zip(
            self._iter_bayesian_modules(self.model_old),
            self._iter_bayesian_modules(self.model),
        ):
            trainer_weight_mu = new_layer.weight_mu
            saver_weight_mu = old_layer.weight_mu

            trainer_bias = getattr(new_layer, "bias", None)
            saver_bias = getattr(old_layer, "bias", None)

            trainer_weight_sigma = new_layer.weight_sigma
            saver_weight_sigma = old_layer.weight_sigma

            fan_in, _ = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            std_init = math.sqrt((2.0 / fan_in) * self.cfg.ratio)

            saver_strength = std_init / (saver_weight_sigma + eps)
            saver_strength_flat = saver_strength.view(saver_strength.size(0), -1)
            bias_strength = saver_strength_flat.mean(dim=1)

            mu_weight_reg = mu_weight_reg + ((saver_strength * (trainer_weight_mu - saver_weight_mu)) ** 2).sum()

            if trainer_bias is not None and saver_bias is not None:
                mu_bias_reg = mu_bias_reg + ((bias_strength * (trainer_bias - saver_bias)) ** 2).sum()

                saver_sigma_flat = saver_weight_sigma.view(saver_weight_sigma.size(0), -1)
                l1_mu_bias_reg = l1_mu_bias_reg + (
                    (saver_bias.pow(2) / (saver_sigma_flat.mean(dim=1).pow(2) + eps))
                    * (trainer_bias - saver_bias).abs()
                ).sum()

            l1_mu_weight_reg = l1_mu_weight_reg + (
                (saver_weight_mu.pow(2) / saver_weight_sigma.pow(2))
                * (trainer_weight_mu - saver_weight_mu).abs()
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

    def _iter_bayesian_modules(self, module: nn.Module) -> Iterable[BayesianLayer]:
        for sub in module.modules():
            if isinstance(sub, BayesianLayer):
                yield sub

    @torch.no_grad()
    def mc_epistemic_classification(self, x, t, S=20, temperature=1.0, clamp_eps=1e-8):
        """Monte-Carlo epistemic uncertainty for classification."""

        model = self.model
        model.eval()
        probs_accum = []

        for _ in range(S):
            logits = model(x, sample=True)[t] / temperature
            probs = F.softmax(logits, dim=-1)
            probs_accum.append(probs)

        probs_stack = torch.stack(probs_accum, dim=0)
        p_mean = probs_stack.mean(dim=0)

        p_mean_clamped = p_mean.clamp(min=clamp_eps, max=1.0)
        H_pred = -(p_mean_clamped * p_mean_clamped.log()).sum(dim=-1)

        probs_clamped = probs_stack.clamp(min=clamp_eps, max=1.0)
        entropies = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        EH = entropies.mean(dim=0)

        MI = H_pred - EH

        return p_mean, H_pred, EH, MI


__all__ = ["Net"]
