# TODO: update to work with ResNet PLN

import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.resnet1d import ResNet1D
from model.modelfactory import ModelFactory
# from dataloaders.iq_data_loader import ensure_iq_two_channel

logger = logging.getLogger("experiment")


@dataclass
class AnmlBaseConfig:
    alpha_init: float = 1e-3

    @staticmethod
    def from_args(args: object) -> "AnmlBaseConfig":
        cfg = AnmlBaseConfig()
        if hasattr(args, "alpha_init"):
            cfg.alpha_init = getattr(args, "alpha_init")
        return cfg

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
    running_var = torch.ones(np.prod(np.array(input.data.size()[1])))
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)
def maxpool1d(input, kernel_size, stride=None):
    return F.max_pool1d(input, kernel_size, stride)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv1d(input, weight, bias, stride, padding, dilation, groups)

class Learner(nn.Module):

    def __init__(self, n_outputs, args, neuromodulation=True):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()
        self.args = args
        self.cfg = AnmlBaseConfig.from_args(args)
        
        # self.config = config
        self.Neuromodulation = neuromodulation
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        self.net = ResNet1D(n_outputs, self.args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.config = ModelFactory.get_model()

        for i, (name, param) in enumerate(self.config):
            if 'conv' in name:
                cout, cin, k = param[:3]       # ensure your config stores 1D conv as [cout, cin, k, (stride), (pad)]
                w = nn.Parameter(torch.empty(cout, cin, k))
                torch.nn.init.kaiming_normal_(w)
                # [ch_out, ch_in, kernelsz]
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(cout)))

            elif 'linear' in name or 'nm_to' in name or name == 'fc':

                # [ch_out, ch_in]
                sz_r, sz_int = param
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif 'bn' in name:
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                
            else:
                raise NotImplementedError

        # # iterate over all parameters
        # for name, param in self.model.named_parameters():
        #     self.vars.append(param)

        # # iterate over buffers (like BN running stats)
        # for name, buf in self.model.named_buffers():
        #     if "running_mean" in name or "running_var" in name:
        #         # match your old code: store them but freeze grads
        #         p = nn.Parameter(buf.clone().detach(), requires_grad=False)
        #         self.vars_bn.append(p)


    def forward(self, x, vars=None, bn_training=True, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        
        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0

        if self.Neuromodulation:

            # =========== NEUROMODULATORY NETWORK ===========

            #'conv1_nm'
            #'bn1_nm'
            #'conv2_nm'
            #'bn2_nm'
            #'conv3_nm'
            #'bn3_nm'

          
            # Query the neuromodulatory network:
            
            
            data = x.view(-1,2,2048)
            nm_data = x.view(-1,2,2048)

            #input_mask = self.call_input_nm(data_, vars)
            #fc_mask = self.call_fc_nm(data_, vars)

            w,b = vars[0], vars[1]
            nm_data = conv1d(nm_data, w, b)
            w,b = vars[2], vars[3]
            running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)

            nm_data = F.relu(nm_data)
            nm_data = maxpool1d(nm_data, kernel_size=2, stride=2)

            w,b = vars[4], vars[5]
            nm_data = conv1d(nm_data, w, b)
            w,b = vars[6], vars[7]
            running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)

            nm_data = F.relu(nm_data)
            nm_data = maxpool1d(nm_data, kernel_size=2, stride=2)

            w,b = vars[8], vars[9]
            nm_data = conv1d(nm_data, w, b)
            w,b = vars[10], vars[11]
            running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)
            nm_data = F.relu(nm_data)
            #nm_data = maxpool(nm_data, kernel_size=2, stride=2)


            nm_data = nm_data.view(nm_data.size(0), -1)

            # NM Output

            w = vars[12]
            b = vars[13]
            fc_mask = F.sigmoid(F.linear(nm_data, w, b))#.view(nm_data.size(0), 512)


            # =========== PREDICTION NETWORK ===========

            prediction = self.net.forward(x, fc_mask)

            try:
                prediction = torch.cat([prediction, data], dim=0)
            except:
                prediction = data
            
            return(prediction)


        else:
            x = self.net.forward(x)
            return (x) 

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuromodNet1D(nn.Module):
    """
    A 1D neuromodulatory network that outputs a per-sample mask.
    Compatible with MAML/ANML: can accept a dict of fast-weights (fw).
    """
    def __init__(self, in_ch: int, mask_dim: int, conv_ch=(64, 112, 112),
                 k=5, pool_every=(1,), stride=1, pad='same'):
        super().__init__()
        C1, C2, C3 = conv_ch

        # --- Conv stack (registered so you can outer-train them) ---
        self.conv1 = nn.Conv1d(in_ch, C1, kernel_size=k, stride=stride,
                               padding=(k//2 if pad=='same' else 0), bias=True)
        self.bn1   = nn.BatchNorm1d(C1, affine=True)

        self.conv2 = nn.Conv1d(C1, C2, kernel_size=k, stride=stride,
                               padding=(k//2 if pad=='same' else 0), bias=True)
        self.bn2   = nn.BatchNorm1d(C2, affine=True)

        self.conv3 = nn.Conv1d(C2, C3, kernel_size=k, stride=stride,
                               padding=(k//2 if pad=='same' else 0), bias=True)
        self.bn3   = nn.BatchNorm1d(C3, affine=True)

        self.pool_every = set(pool_every)  # layers after which to pool (e.g., {2})
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- We’ll set fc after we know in_features via dummy pass ---
        self.fc = None
        self.mask_dim = mask_dim

    # helpers to pick fast-weights if provided
    def _W(self, module, attr, fw):
        if fw is None: return getattr(module, attr)
        key = f'{module._get_name()}.{id(module)}.{attr}'
        return fw.get(key, getattr(module, attr))

    def _bn(self, bn_mod, x, fw):
        w = self._W(bn_mod, 'weight', fw)
        b = self._W(bn_mod, 'bias', fw)
        # use running stats from module (fast BN stats are rarely needed)
        return F.batch_norm(x, bn_mod.running_mean, bn_mod.running_var,
                            weight=w, bias=b, training=self.training)

    def features_only(self, x, fw=None):
        # x: (B, Cin, L)
        x = F.conv1d(x, self._W(self.conv1, 'weight', fw), self._W(self.conv1, 'bias', fw),
                     stride=self.conv1.stride, padding=self.conv1.padding, dilation=self.conv1.dilation, groups=1)
        x = self._bn(self.bn1, x, fw); x = F.relu(x)
        if 1 in self.pool_every: x = self.pool(x)

        x = F.conv1d(x, self._W(self.conv2, 'weight', fw), self._W(self.conv2, 'bias', fw),
                     stride=self.conv2.stride, padding=self.conv2.padding, dilation=self.conv2.dilation, groups=1)
        x = self._bn(self.bn2, x, fw); x = F.relu(x)
        if 2 in self.pool_every: x = self.pool(x)

        x = F.conv1d(x, self._W(self.conv3, 'weight', fw), self._W(self.conv3, 'bias', fw),
                     stride=self.conv3.stride, padding=self.conv3.padding, dilation=self.conv3.dilation, groups=1)
        x = self._bn(self.bn3, x, fw); x = F.relu(x)
        if 3 in self.pool_every: x = self.pool(x)

        return x  # (B, C3, L_out)

    def init_fc_if_needed(self, x_sample):
        if self.fc is not None: return
        with torch.no_grad():
            feats = self.features_only(x_sample)           # (1, C3, L_out)
            in_feat = feats.view(1, -1).size(1)
        self.fc = nn.Linear(in_feat, self.mask_dim, device=x_sample.device)

    def forward(self, x, fw=None):
        """
        Returns mask in (B, mask_dim), sigmoid in [0,1].
        If fw is provided (dict), uses those tensors as fast-weights.
        """
        if self.fc is None:
            # one-sample dummy to infer fc in_features
            dummy = x[:1]
            self.init_fc_if_needed(dummy)

        feats = self.features_only(x, fw)           # (B, C3, Lout)
        flat  = feats.view(feats.size(0), -1)
        W = self._W(self.fc, 'weight', fw) if fw else self.fc.weight
        b = self._W(self.fc, 'bias', fw) if fw else self.fc.bias
        mask = torch.sigmoid(F.linear(flat, W, b))  # (B, mask_dim)
        return mask
