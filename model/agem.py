### This is a pytorch implementation of AGEM based on https://github.com/facebookresearch/agem. 

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import random

from model.resnet1d import ResNet1D
from model.detection_replay import DetectionReplayMixin
from utils.training_metrics import macro_recall
from utils import misc_utils


@dataclass
class AgemConfig:
    ## AGEM-specific hyperparameters
    lr: float = 1e-3
    glances: int = 1
    memories: int = 5120

    ## Generic hyperparameters
    arch: str = "resnet1d"
    n_layers: int = 2
    n_hiddens: int = 100
    dataset: str = "tinyimagenet"
    cuda: bool = True
    grad_clip_norm: Optional[float] = 100.0
    input_channels: int = 1
    det_lambda: float = 1.0
    cls_lambda: float = 1.0
    det_memories: int = 2000
    det_replay_batch: int = 64

    @staticmethod
    def from_args(args: object) -> "AgemConfig":
        cfg = AgemConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg
# Auxiliary functions useful for AGEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    return misc_utils.compute_offsets(task, nc_per_task)


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1



def projectgrad(gradient, memories, margin=0.5, eps = 1e-3, oiter = 0):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """

    similarity = torch.nn.functional.cosine_similarity(gradient.t(), memories.t().mean(dim=0).unsqueeze(0))

    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

    # merge memories
    t = memories_np.shape[0]

    memories_np2 = memories_np.mean(axis=0).reshape(1, memories_np.shape[1])

    ref_mag = np.dot(memories_np2, memories_np2.transpose())
    dotp = np.dot(gradient_np.reshape(1, -1), memories_np2.transpose())

    if(oiter%100==0):
        print('similarity : ', similarity.item())
        print('dotp:', dotp)

    if(dotp[0,0]<0):
        proj = gradient_np.reshape(1, -1) - ((dotp/ ref_mag) * memories_np2)
        gradient.copy_(torch.Tensor(proj).view(-1, 1))


class Net(DetectionReplayMixin, nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.cfg = AgemConfig.from_args(args)

        self.is_cifar = (
            (self.cfg.dataset == 'cifar100') or (self.cfg.dataset == 'tinyimagenet')
        )
        
        # --- IQ mode toggle ---
        self.input_channels = self.cfg.input_channels
        self.is_iq = (self.cfg.dataset == "iq") or (self.input_channels == 2)

        if self.cfg.arch != 'resnet1d':
            raise ValueError(f"Unsupported arch {self.cfg.arch}; only resnet1d is available now.")
        self.net = ResNet1D(n_outputs, args)

        self.ce = nn.CrossEntropyLoss()
        self.bce = torch.nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.glances = self.cfg.glances
        self.det_lambda = float(self.cfg.det_lambda)
        self.cls_lambda = float(self.cfg.cls_lambda)
        self._init_det_replay(
            self.cfg.det_memories,
            self.cfg.det_replay_batch,
            enabled=bool(getattr(args, "use_detector_arch", False)),
        )

        self.opt = optim.SGD(self._ll_params(), self.cfg.lr, momentum=0.9)
        self.det_opt = optim.SGD(self.net.det_head.parameters(), self.cfg.lr, momentum=0.9)

        self.n_memories = int(self.cfg.memories/n_tasks)
        self.gpu = self.cfg.cuda

        self.age = 0
        self.M = []
        self.memories = self.cfg.memories
        self.grad_align = []
        self.grad_task_align = {}
        self.current_task = None

        # --- Episodic memory allocation ---
        if self.is_iq:
            assert n_inputs % 2 == 0, f"n_inputs={n_inputs} must be 2*L for IQ."
            self.seq_len = n_inputs // 2
            # (task, mem, C=2, L)
            self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, 2, self.seq_len)
        else:
            # (task, mem, F)
            self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)

        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self._ll_params():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.mem_cnt = 0
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "") or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        
        if self.gpu:
            self.cuda()

        self.iter = 0

    def _ensure_iq_shape(self, x):
        """
        Ensure x is (B, 2, L) for IQ mode.
        Accepts (B, 2, L) or (B, 2L).
        """
        if x.dim() == 3:
            # (B, 2, L) already
            return x
        elif x.dim() == 2:
            # (B, 2L) -> (B, 2, L)
            B, F = x.shape
            assert F % 2 == 0, f"Feature dim {F} not divisible by 2 for (2, L) reshape."
            L = F // 2
            return x.view(B, 2, L)
        else:
            raise ValueError(f"Unexpected IQ input shape {tuple(x.shape)}; expected (B, 2, L) or (B, 2L).")

    def _adapt_for_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure inputs stored in memory are (B, 2, L)."""
        if x.dim() == 4 and x.size(1) == 3 and x.size(2) == 2:
            return self.net.model.input_adapter(x)
        if x.dim() == 3 and x.size(1) == 3:
            if x.size(2) % 2 != 0:
                raise ValueError(
                    f"Expected even length for 3-ADC IQ input; got shape {tuple(x.shape)}."
                )
            seq_len = x.size(2) // 2
            x4 = x.view(x.size(0), 3, 2, seq_len)
            return self.net.model.input_adapter(x4)
        if x.dim() == 2:
            features = x.size(1)
            if features % 6 == 0:
                seq_len = features // 6
                x4 = x.view(x.size(0), 3, 2, seq_len)
                return self.net.model.input_adapter(x4)
        return self._ensure_iq_shape(x)
    
    def forward(self, x, t):
        if self.cfg.dataset == 'tinyimagenet':
            x = x.view(-1, 3, 64, 64)
        elif self.cfg.dataset == 'cifar100':
            x = x.view(-1, 3, 32, 32)
        elif self.is_iq:
            x = self._ensure_iq_shape(x) # (B, 2, L)

        output = self.net.forward(x)

        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = compute_offsets(t, self.classes_per_task, self.is_cifar)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def _ll_params(self):
        for name, param in self.net.named_parameters():
            if name.startswith("det_head"):
                continue
            yield param

    def observe(self, x, y, t):

        self.iter +=1
        
        # --- shape handling ---
        if self.is_iq:
            # keep (B, 2, L)
            x = self._ensure_iq_shape(x)
        else:
            # legacy: flatten non-IQ inputs
            x = x.view(x.size(0), -1)
        class_counts = getattr(self, "classes_per_task", None)
        noise_label = None
        if class_counts is not None:
            _, offset2 = misc_utils.compute_offsets(t, class_counts)
            noise_label = offset2 - 1
        y_cls, y_det = self._unpack_labels(
            y,
            noise_label=noise_label,
            use_detector_arch=bool(getattr(self, "det_enabled", False)),
        )
        if y_det is not None and self.det_memories > 0:
            self._update_det_memory(x, y_det)
        x_det = x
        signal_mask = (y_det == 1) & (y_cls >= 0)
        if not signal_mask.any():
            self.det_opt.zero_grad()
            det_logits, _ = self.net.forward_heads(x_det)
            det_loss = self.det_loss(det_logits, y_det.float())
            det_replay = self._sample_det_memory()
            if det_replay is not None:
                mem_x, mem_y = det_replay
                mem_det_logits, _ = self.net.forward_heads(mem_x)
                mem_loss = self.det_loss(mem_det_logits, mem_y.float())
                det_loss = 0.5 * (det_loss + mem_loss)
            det_loss = self.det_lambda * det_loss
            det_loss.backward()
            self.det_opt.step()
            return float(det_loss.item()), 0.0

        x = x[signal_mask]
        y = y_cls[signal_mask]

        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            self.grad_align.append([])
            
        tr_acc = []
        for pass_itr in range(self.glances):
# copy x into memory with matching shape
                
            if(pass_itr==0):
                # Update ring buffer storing examples from current task
                bsz = y.data.size(0)
                endcnt = min(self.mem_cnt + bsz, self.n_memories)
                effbsz = endcnt - self.mem_cnt
                # self.memory_data[t, self.mem_cnt: endcnt].copy_(
                #     x.data[: effbsz])
                # if bsz == 1:
                #     self.memory_labs[t, self.mem_cnt] = y.data[0]
                # else:
                #     self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                #         y.data[: effbsz])
                # self.mem_cnt += effbsz
                # if self.mem_cnt == self.n_memories:
                #     self.mem_cnt = 0

                if effbsz > 0:
                    # mem_x = self._adapt_for_memory(x.data[:effbsz])
                    self.memory_data[t, self.mem_cnt:endcnt].copy_(x.data[:effbsz])

                    if bsz == 1:
                        self.memory_labs[t, self.mem_cnt] = y.data[0]
                    else:
                        self.memory_labs[t, self.mem_cnt:endcnt].copy_(y.data[:effbsz])

                    self.mem_cnt += effbsz
                    if self.mem_cnt == self.n_memories:
                        self.mem_cnt = 0

            # compute gradient on previous tasks
            if len(self.observed_tasks) > 1:
                for tt in range(len(self.observed_tasks) - 1):
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]

                    offset1, offset2 = compute_offsets(past_task, self.classes_per_task,
                                                       self.is_cifar)
                    logits = self.forward(Variable(self.memory_data[past_task]), past_task)[:, offset1: offset2]
                    ptloss = self.ce(
                        logits,
                        Variable(self.memory_labs[past_task] - offset1))
                    ptloss.backward()
                    if self.cfg.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

                    store_grad(self._ll_params, self.grads, self.grad_dims,
                               past_task)

            # now compute the grad on the current minibatch
            self.zero_grad()
            offset1, offset2 = compute_offsets(t, self.classes_per_task, self.is_cifar)
            logits = self.forward(x, t)[:, offset1: offset2]
            pb = torch.argmax(logits, dim=1)
            targets = y - offset1
            tr_acc.append(macro_recall(pb, targets))
            loss = self.ce(logits, targets)
            loss.backward()
            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip_norm)

            # check if gradient violates constraints                                                           
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self._ll_params, self.grads, self.grad_dims, t)
                # Build index tensor on the same device as stored gradients.
                indx_device = self.grads.device if hasattr(self.grads, "device") else None
                indx = torch.tensor(
                    self.observed_tasks[:-1],
                    dtype=torch.long,
                    device=indx_device,
                )

                projectgrad(self.grads[:, t].unsqueeze(1),                                           
                              self.grads.index_select(1, indx), oiter = self.iter)
                # copy gradients back
                overwrite_grad(self._ll_params, self.grads[:, t],
                               self.grad_dims)

            self.opt.step()
        
        xi = x.data.cpu().numpy()
        yi = y.data.cpu().numpy()
        for i in range(0,x.size()[0]):
            self.age += 1
            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi[i],yi[i],t])

            else:
                p = random.randint(0,self.age)
                if p < self.memories:
                    self.M[p] = [xi[i],yi[i],t]

        self.det_opt.zero_grad()
        det_logits, _ = self.net.forward_heads(x_det)
        det_loss = self.det_loss(det_logits, y_det.float())
        det_replay = self._sample_det_memory()
        if det_replay is not None:
            mem_x, mem_y = det_replay
            mem_det_logits, _ = self.net.forward_heads(mem_x)
            mem_loss = self.det_loss(mem_det_logits, mem_y.float())
            det_loss = 0.5 * (det_loss + mem_loss)
        det_loss = self.det_lambda * det_loss
        det_loss.backward()
        self.det_opt.step()

        avg_tr_acc = sum(tr_acc)/len(tr_acc) if tr_acc else 0.0
        return loss.item(), avg_tr_acc
