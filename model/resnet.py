import torch
import torch.nn as nn
from torchvision.models import resnet18


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

    def forward(self, x):
        return self.model(x)

    def define_task_lr_params(self, alpha_init=1e-3):
        self.alpha_lr = nn.ParameterList([])
        for p in self.parameters():
            self.alpha_lr.append(nn.Parameter(torch.ones_like(p) * alpha_init))

