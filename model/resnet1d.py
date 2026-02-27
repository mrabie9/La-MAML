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


class AdcIqAdapter(nn.Module):
    """Reduce 3 ADC channels into 2 IQ channels.

    Expects input of shape (B, 3, 2, L) and returns (B, 2, L).
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 3))
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3 or x.size(2) != 2:
            raise ValueError(
                f"ADC adapter expects (B, 3, 2, L); got shape {tuple(x.shape)}."
            )
        # (B, 3, 2, L) -> (B, 2, 3, L)
        x = x.permute(0, 2, 1, 3)
        # Mix ADCs per IQ channel: (B, 2, 3, L) x (2, 3) -> (B, 2, L)
        y = torch.einsum("bial,ia->bil", x, self.weight)
        y = y + self.bias.view(1, 2, 1)
        return y


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

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.drop1 = nn.Dropout(p=0.2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.drop2 = nn.Dropout(p=0.2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.drop3 = nn.Dropout(p=0.2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.drop4 = nn.Dropout(p=0.2)

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
            if x.dim() == 4:
                if not isinstance(self.input_adapter, nn.Identity):
                    # print("Using input adapter for 4D input.", x.shape)
                    x = self.input_adapter(x)
                    # print("Adapter output shape:", x.shape)
                else:
                    raise ValueError("Received 4D input but no adapter is configured.")

            x = self.bn1(self.conv1(x))
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.drop1(x)
            x = self.layer2(x)
            x = self.drop2(x)
            x = self.layer3(x)
            x = self.drop3(x)
            x = self.layer4(x)
            x = self.drop4(x)

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
        self.input_adapter = AdcIqAdapter()
        self.model = _ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            num_classes,
            in_channels=in_channels,
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
                # print(f"Input shape: {tuple(x.shape)}")
                x = self._prepare_input(x)
                # print(f"Prepared input shape: {tuple(x.shape)}")
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
        """Normalize inputs into channel-first IQ tensors.

        Accepted formats:
        - (B, F): flat samples. If F is divisible by 2 or 3, reshape to
          (B, 2, L) or (B, 3, 2, L) respectively. Ambiguous shapes (divisible
          by both 2 and 3) raise an error.
        - (B, C, L): channel-first. C can be 1 or 2 (passed through) or 3
          (interpreted as 3 ADC channels with interleaved IQ and reshaped to
          (B, 3, 2, L)).
        - (C, B, L): channel-first but transposed; will be permuted to (B, C, L).
        - (B, 3, 2, L): explicit ADC + IQ layout (passed through).
        """
        if x.dim() == 2:
            batch, features = x.shape
            if features % 2 == 0 and features % 3 != 0:
                seq_len = features // 2
                x = x.view(batch, 2, seq_len)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[1] not in (1, 2, 3) and x.shape[0] in (1, 2, 3):
                x = x.permute(1, 0, 2).contiguous()
            if x.shape[1] == 3:
                if x.shape[2] % 2 != 0:
                    raise ValueError(
                        f"Expected even length for 3-ADC IQ input; got shape {tuple(x.shape)}."
                    )
                seq_len = x.shape[2] // 2
                x = x.view(x.shape[0], 3, 2, seq_len)
                return x
            if x.shape[1] not in (1, 2):
                raise ValueError(
                    f"Unexpected channel dimension (expected 1, 2, or 3); got shape {tuple(x.shape)}."
                )
        elif x.dim() == 4:
            if x.shape[1] == 3 and x.shape[2] == 2:
                return x
            raise ValueError(
                f"Unexpected 4D input shape {tuple(x.shape)}; expected (B, 3, 2, L)."
            )
        else:
            raise ValueError(
                f"Unexpected input shape {tuple(x.shape)}; expected 2D, 3D, or 4D tensor."
            )
        return x

    # ------------------------------------------------------------------
    def _build_norm_factory(self, args):
        # use_groupnorm = getattr(args, "model", "") == "packnet"
        # if use_groupnorm:
        #     target_groups = getattr(args, "groupnorm_groups", 16)

        #     def gn_factory(channels: int):
        #         groups = max(1, math.gcd(channels, target_groups))
        #         return nn.GroupNorm(groups, channels)

        #     return gn_factory
        return lambda c: nn.BatchNorm1d(c)

__all__ = ["ResNet1D"]
