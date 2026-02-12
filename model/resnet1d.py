"""1D ResNet-18 architecture for raw IQ data.

This module implements a lightweight 1D variant of the popular ResNet-18
architecture.  The network operates on two input channels representing the
in-phase and quadrature components of complex signals and is thus suitable for
raw IQ sample classification tasks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.func import functional_call


class BasicBlock1D(nn.Module):
    """1D version of the standard ResNet ``BasicBlock``."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None, norm_layer=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover -
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class _ResNet1D(nn.Module):
    """Internal utility that mirrors ``torchvision``'s ``ResNet`` logic."""

    def __init__(self, block: type[nn.Module], layers: list[int], num_classes: int,
                 in_channels: int = 2, norm_layer=None, input_adapter: nn.Module | None = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.input_adapter = input_adapter or nn.Identity()

        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=1, bias=False
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        out_dim = 512 * block.expansion
        self.fc = nn.Linear(out_dim, num_classes)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=stride, bias=False
                ),
                self._norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features=False, classify_feats=False, return_h4=False) -> torch.Tensor:  # pragma: no cover -
        
        if classify_feats:
            return self.fc(x)
        else:
            if not isinstance(self.input_adapter, nn.Identity) and x.size(1) == 3:
                x = self.input_adapter(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            if return_h4:
                return x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if return_features:
                return x
            x = self.fc(x)
            return x


class ResNet1D(nn.Module):
    """ResNet-18 style network using 1D convolutions for IQ data."""

    def __init__(self, num_classes: int, args=None, in_channels: int = 2) -> None:
        super().__init__()
        self.args = args
        norm_layer = self._build_norm_factory(args)
        if in_channels == 3:
            self.input_adapter = nn.Conv1d(3, 1, kernel_size=1, bias=False)
            backbone_in = 1
        else:
            self.input_adapter = nn.Identity()
            backbone_in = in_channels
        self.model = _ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            num_classes,
            in_channels=backbone_in,
            norm_layer=norm_layer,
            input_adapter=self.input_adapter,
        )
        self.feature_dim = self.model.fc.in_features
        self.det_head = nn.Linear(self.feature_dim, 1)

        # Ordered names for mapping fast weights
        self.param_names = [n for n, _ in self.model.named_parameters()]
        # Learner-compat convenience (list, not registered)
        self.vars = list(self.model.parameters())
        alpha_init = getattr(args, "alpha_init", 1e-3) if args is not None else 1e-3
        self.alpha_lr = nn.ParameterList(
            [nn.Parameter(torch.ones_like(p) * alpha_init) for p in self.model.parameters()]
        )

    # def classify_feats(self, x: torch.Tensor) -> torch.Tensor:
    #     logits = self.model.fc(x)
    #     return logits

    def forward(self, x: torch.Tensor, vars=None, bn_training: bool = True, classify_feats=False, ret_feats = False) -> torch.Tensor:
        prev = self.model.training
        self.model.train(bn_training)
        try:
            if not classify_feats:
                x = self._prepare_input(x)
            if vars is None:
                out = self.model(x, return_features=ret_feats, classify_feats=classify_feats)
            else:
                assert len(vars) == len(self.param_names), (
                    f"len(vars)={len(vars)} vs params={len(self.param_names)}"
                )
                param_dict = {n: p for n, p in zip(self.param_names, vars)}
                out = functional_call(
                    self.model,
                    param_dict,
                    (x,),
                    {"return_features": ret_feats, "classify_feats": classify_feats},
                )
        finally:
            self.model.train(prev)
        return out

    def forward_features(self, x: torch.Tensor, vars=None, bn_training: bool = True) -> torch.Tensor:
        return self.forward(x, vars=vars, bn_training=bn_training, ret_feats=True)

    def forward_classifier(self, feats: torch.Tensor, vars=None, bn_training: bool = True) -> torch.Tensor:
        return self.forward(feats, vars=vars, bn_training=bn_training, classify_feats=True)

    def forward_detection(self, feats: torch.Tensor) -> torch.Tensor:
        return self.det_head(feats).squeeze(1)

    def forward_heads(self, x: torch.Tensor, vars=None, bn_training: bool = True):
        feats = self.forward_features(x, vars=vars, bn_training=bn_training)
        det_logits = self.forward_detection(feats)
        cls_logits = self.forward_classifier(feats, vars=vars, bn_training=bn_training)
        return det_logits, cls_logits

    # Expose only the underlying model parameters, excluding alpha lrs
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def define_task_lr_params(self, alpha_init: float = 1e-3) -> None:
        self.alpha_lr = nn.ParameterList(
            [nn.Parameter(torch.ones_like(p) * alpha_init) for p in self.model.parameters()]
        )

    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 != 0:
                seq_len = features // 2
                x = x.view(batch, 2, seq_len)
            elif features % 3 == 0 and features % 2 != 0:
                seq_len = features // 3
                x = x.view(batch, 3, seq_len)
            elif features % 2 == 0 and features % 3 == 0:
                raise ValueError(
                    f"Ambiguous flat input shape {tuple(x.shape)}; divisible by both 2 and 3. "
                    "Provide an explicit (B, C, L) tensor with C=2 or C=3."
                )
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[1] not in (1, 2, 3) and x.shape[0] in (1, 2, 3):
                x = x.permute(1, 0, 2).contiguous()
            if x.shape[1] not in (1, 2, 3):
                raise ValueError(
                    f"Unexpected channel dimension (expected 1, 2, or 3); got shape {tuple(x.shape)}."
                )
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}; expected 2D or 3D tensor.")
        return x

    # ------------------------------------------------------------------
    def _build_norm_factory(self, args):
        use_groupnorm = getattr(args, "model", "") == "packnet"
        if use_groupnorm:
            target_groups = getattr(args, "groupnorm_groups", 4)

            def gn_factory(channels: int):
                groups = max(1, math.gcd(channels, target_groups))
                return nn.GroupNorm(groups, channels)

            return lambda c: nn.Identity(c) # gn_factory
        return lambda c: nn.BatchNorm1d(c)

__all__ = ["ResNet1D"]
