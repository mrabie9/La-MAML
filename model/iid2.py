from dataclasses import dataclass
import sys

import torch

from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("once")


@dataclass
class IidConfig:
    """Configuration shim for IID (non-LL) runs.

    This mirrors the basic argument harvesting pattern used by other models
    while keeping the training behaviour strictly non-lifelong.

    Attributes:
        arch: Backbone architecture identifier (only ``\"resnet1d\"`` is supported).
        lr: Learning rate for the SGD optimizer.
        cuda: Whether to place the model on GPU.

    Usage:
        cfg = IidConfig.from_args(args)
    """

    arch: str = "resnet1d"
    lr: float = 1e-3
    cuda: bool = True

    @staticmethod
    def from_args(args: object) -> "IidConfig":
        cfg = IidConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(torch.nn.Module):
    """Plain ResNet1D classifier for non-lifelong (IID) experiments.

    This model shares the same ``ResNet1D`` backbone as the other approaches
    in the repository but **does not** implement any continual-learning
    machinery (no task-wise offsets, no memory, no regularisers).

    The ``t`` argument is accepted for API compatibility with the training
    loops but is ignored during the forward pass and optimisation.

    Usage:
        model = Net(n_inputs, n_outputs, n_tasks, args)
        logits = model(x, t)          # t is ignored
        loss, rec = model.observe(x, y, t)
    """

    def __init__(self, n_inputs: int, n_outputs: int, n_tasks: int, args: object) -> None:
        super().__init__()
        del n_inputs  # ResNet1D determines its own front-end shape
        del n_tasks

        self.cfg = IidConfig.from_args(args)
        self.n_outputs = n_outputs

        if self.cfg.arch != "resnet1d":
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")

        # Shared backbone with other models.
        self.net = ResNet1D(n_outputs, args)

        # Optimiser and loss.
        self.opt = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, t: torch.Tensor | int) -> torch.Tensor:  # pragma: no cover - thin wrapper
        """Return logits for all classes, ignoring task information."""
        del t
        return self.net(x)

    def observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | int,
    ) -> tuple[float, float]:
        """Perform a single SGD step on IID data.

        Args:
            x: Input batch.
            y: Ground-truth class labels (0..n_outputs-1).
            t: Task index (ignored; present only for interface compatibility).

        Returns:
            Tuple of (loss_value, training_recall).

        Usage:
            loss, rec = model.observe(x, y, t)
        """
        del t

        self.train()
        self.opt.zero_grad()

        logits = self.net(x)
        targets = y.long()

        loss_tensor = self.loss(logits, targets)
        loss_tensor.backward()
        self.opt.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            cls_tr_rec = macro_recall(preds.detach().cpu(), targets.detach().cpu())

        return float(loss_tensor.item()), float(cls_tr_rec)


__all__ = ["Net"]

