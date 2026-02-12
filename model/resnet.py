import torch
import torch.nn as nn
from torchvision.models import resnet18


from torch.func import functional_call  # use this on torch 1.12.x

class ResNet18(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.args = args
        self.model = resnet18(pretrained=False)

        if getattr(args, "dataset", None) in ["cifar100", "tinyimagenet"]:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Ordered names for mapping fast weights:
        self.param_names = [n for n, _ in self.model.named_parameters()]
        # Learner-compat convenience (list, not registered):
        self.vars = list(self.model.parameters())
        # per-parameter inner LRs (if you use them)
        self.alpha_lr = nn.ParameterList([nn.Parameter(torch.ones_like(p) * getattr(args, "alpha_init", 1e-3))
                                          for p in self.model.parameters()])

    def forward(self, x, vars=None, bn_training=True, feature=False):
        prev = self.model.training
        self.model.train(bn_training)
        try:
            if vars is None:
                out = self.model(x)
            else:
                assert len(vars) == len(self.param_names), \
                    f"len(vars)={len(vars)} vs params={len(self.param_names)}"
                param_dict = {n: p for n, p in zip(self.param_names, vars)}
                out = functional_call(self.model, param_dict, (x,))
        finally:
            self.model.train(prev)
        return out


    # Expose only the underlying model parameters, excluding alpha lrs
    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def define_task_lr_params(self, alpha_init=1e-3):
        self.alpha_lr = nn.ParameterList([])
        for p in self.model.parameters():
            self.alpha_lr.append(nn.Parameter(torch.ones_like(p) * alpha_init))

