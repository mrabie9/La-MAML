import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.utils import stateless


class ResNet18(nn.Module):
    """Wrapper around torchvision's ResNet18 with per-parameter learning rates."""

    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args
        self.model = resnet18(pretrained=False)

        # Adjust the first layers for smaller images if needed
        if args.dataset in ["cifar100", "tinyimagenet"]:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Cache parameter names to align with fast-weight lists
        self.param_names = [name for name, _ in self.model.named_parameters()]

    def forward(self, x, params=None):
        """Forward with optional fast weights.

        Args:
            x: input tensor
            params: list or iterable of tensors matching model parameters
        """
        if params is None:
            return self.model(x)
        # Map provided tensors to parameter names and run stateless functional call
        param_dict = {n: p for n, p in zip(self.param_names, params)}
        return stateless.functional_call(self.model, param_dict, (x,), strict=False)

    # Expose only the underlying model parameters, excluding alpha lrs
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def define_task_lr_params(self, alpha_init=1e-3):
        self.alpha_lr = nn.ParameterList([])
        for p in self.parameters():
            self.alpha_lr.append(nn.Parameter(torch.ones_like(p) * alpha_init))

