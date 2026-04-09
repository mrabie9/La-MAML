from dataclasses import dataclass
import sys

import torch

from model.resnet1d import ResNet1D
from model.detection_replay import (
    noise_label_from_args,
    signal_mask_exclude_noise,
    unpack_y_to_class_labels,
)
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

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

    def __init__(
        self, n_inputs: int, n_outputs: int, n_tasks: int, args: object
    ) -> None:
        super().__init__()
        del n_inputs  # ResNet1D determines its own front-end shape

        if n_tasks <= 0:
            raise ValueError("IID2 requires a positive number of tasks")

        self.cfg = IidConfig.from_args(args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
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

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )

        # Shared backbone with other models.
        self.net = ResNet1D(n_outputs, args)

        # Optimiser and loss.
        self.opt = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | int,
        **kwargs,
    ) -> torch.Tensor:  # pragma: no cover - thin wrapper
        """Return logits for all classes; ``t`` is ignored except for API parity.

        For class-incremental evaluation, ``cil_all_seen_upto_task`` masks logits
        for classes not yet introduced (standard CIL protocol).
        """
        del t
        logits = self.net(x)
        cil = kwargs.get("cil_all_seen_upto_task")
        if cil is None:
            return logits
        return misc_utils.apply_task_incremental_logit_mask(
            logits,
            0,
            self.classes_per_task,
            self.n_outputs,
            cil_all_seen_upto_task=cil,
            global_noise_label=self.noise_label,
            loader=self.incremental_loader_name,
        )

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
        targets = unpack_y_to_class_labels(y).long()
        signal_mask = signal_mask_exclude_noise(targets, self.noise_label)
        loss_tensor = classification_cross_entropy(
            logits,
            targets,
            class_weighted_ce=self.class_weighted_ce,
        )
        loss_tensor.backward()
        self.opt.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            if signal_mask.any():
                cls_tr_rec = macro_recall(
                    preds[signal_mask].detach().cpu(),
                    targets[signal_mask].detach().cpu(),
                )
            else:
                cls_tr_rec = 0.0

        return float(loss_tensor.item()), float(cls_tr_rec)


__all__ = ["Net"]
