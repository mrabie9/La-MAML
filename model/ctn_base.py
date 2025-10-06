# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d, normalize
import numpy as np
import pdb
from torch.nn.utils import weight_norm as wn
from itertools import chain
from model.resnet1d import _ResNet1D, BasicBlock1D

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)
       
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def Flatten(x):
    return x.view(x.size(0), -1)

class noReLUBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(noReLUBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class ContextNet(nn.Module):
    def __init__(self, num_classes, in_channels = 2, task_emb=64 , n_tasks = 17):
        super(ContextNet, self).__init__()
        self.in_planes = nf = 64
        # self.conv1 = conv3x3(3, nf * 1)
        # self.bn1 = nn.BatchNorm2d(nf * 1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.model = _ResNet1D(
            BasicBlock1D, [2, 2, 2, 2], num_classes, in_channels=in_channels
        )
        
        #self.film1 = nn.Linear(task_emb, nf * 1 * 2)
        #self.film2 = nn.Linear(task_emb, nf * 2 * 2)
        #self.film3 = nn.Linear(task_emb, nf * 4 * 2)
        self.film4 = nn.Linear(task_emb, nf * 8 * 2)
        self.nf = nf
        self.emb = torch.nn.Embedding(n_tasks, task_emb)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def base_param(self):
        base_iter = chain(self.model.conv1.parameters(), self.model.bn1.parameters(),
                    self.model.layer1.parameters(), self.model.layer2.parameters(), self.model.layer3.parameters(),
                    self.model.layer4.parameters(), self.model.fc.parameters())
        for param in base_iter:
            yield param
    
    def context_param(self):
        film_iter = chain(self.emb.parameters(), self.film4.parameters())
        for param in film_iter:
            yield param

    def forward(self, x, t, use_all = True):
        # tmp = torch.LongTensor(t+1)
        # t = tmp.repeat(x.size(0)).cuda()
        device = x.device
        B = x.size(0)

        if isinstance(t, int):
            # one task id for the whole batch
            t = torch.full((B,), t, dtype=torch.long, device=device)
        else:
            # list/tuple/tensor → tensor on the right device/dtype
            t = torch.as_tensor(t, dtype=torch.long, device=device)
            if t.ndim == 0 or t.numel() == 1:
                # scalar or single-element → broadcast
                t = t.view(1).expand(B)
            else:
                t = t.view(-1)
                if t.numel() != B:
                    raise ValueError(f"Expected {B} task ids, got {t.numel()}.")
        t = self.emb(t)
        # bsz = x.size(0)
        # if x.dim() < 4:
        #     x = x.view(bsz,3,32,32)
        h4 = self.model.forward(x, return_h4=True)
        # h0 = self.conv1(x)
        # h0 = relu(self.bn1(h0))
        # h0  = self.maxpool(h0)
        # h1 = self.layer1(h0)
        # h2 = self.layer2(h1)
        # h3 = self.layer3(h2)
        # h4 = self.layer4(h3)

        B, C, L = h4.shape
        film4 = self.film4(t)              # [B, 2*C]
        gamma4, beta4 = film4.split(C, 1)   # [B, C], [B, C]
        gamma4 = normalize(gamma4, p=2, dim=1).view(B, C, 1)
        beta4  = normalize(beta4,  p=2, dim=1).view(B, C, 1)
        h4_new = gamma4 * h4 + beta4        # -> [B, C, L]

        # B, C, L = h4.shape
        # film4 = self.film4(t)
        # gamma4, beta4 = film4.split(C, 1)
        # # gamma4 = film4[:, :self.nf*8]#.view(film4.size(0),-1,1,1)
        # # beta4 = film4[:, self.nf*8:]#.view(film4.size(0),-1,1,1)
        # gamma_norm = gamma4.norm(p=2, dim=1, keepdim = True).view(B, C, 1).detach()
        # beta_norm = beta4.norm(p=2, dim=1, keepdim= True).view(B, C, 1).detach()
        
        # gamma4 = gamma4.div(gamma_norm).view(film4.size(0), -1,1,1) 
        # beta4 = beta4.div(beta_norm).view(film4.size(0), -1, 1, 1)
        # temp = gamma4*h4
        # h4_new = temp + beta4
        
        if use_all:
            h4 = relu(h4_new) + relu(h4)
        else:
            h4 = relu(h4_new)

        out = self.model.avgpool(h4)
        feat = out.view(out.size(0), -1)
        y = self.model.fc(feat)       
        return y

def ContextNet18(num_classes, n_tasks = 17, task_emb = 64):
    return ContextNet(num_classes, n_tasks = n_tasks, task_emb = task_emb)
