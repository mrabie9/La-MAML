import random
import numpy as np
import ipdb
import math

import torch
import torch.nn as nn
from model.lamaml_base import *


class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args):
        super(Net, self).__init__(n_inputs,
                                 n_outputs,
                                 n_tasks,           
                                 args)

        self.nc_per_task = n_outputs

    def forward(self, x, t):
        output = self.net.forward(x)
        return output

    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        logits = self.net.forward(x, fast_weights)
        loss_q = self.loss(logits.squeeze(1), y)
        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        logits = self.net.forward(x, fast_weights)
        loss = self.loss(logits, y)   

        if fast_weights is None:
            fast_weights = list(self.net.parameters())

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)
        # grads = [g.clamp(min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm) for g in grads]
        # for i in range(len(grads)):
        #     torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        # fast_weights = list(
        #         map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))
        
        fast_weights = [
            w.detach() - g.detach() * nn.functional.relu(a)   # depends on a
            for (g, (w, a)) in zip(grads, zip(fast_weights, self.net.alpha_lr))
            ]

        # fast_weights = [w.detach() for w in fast_weights]
        return fast_weights

    def observe(self, x, y, t):
        self.net.train() 

        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr
            
            # shuffle the data (again)
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]
            
            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new
                self.current_task = t

            batch_sz = x.shape[0]
            # meta_losses = [0 for _ in range(batch_sz)] 

            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            fast_weights = None
            
            self.zero_grads()
            for i in range(0, batch_sz):
                batch_x = x[i].unsqueeze(0)
                batch_y = y[i].unsqueeze(0)

                fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t) # Update task-specific fast weights
                # if(self.real_epoch == 0):
                #     self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                if self.real_epoch == 0: # always true
                    with torch.no_grad():
                        self.push_to_mem(batch_x.detach().cpu(), batch_y.detach().cpu(), torch.tensor(t))

                meta_loss, _ = self.meta_loss(bx, fast_weights, by, t) # loss on the meta batch
                # meta_losses[i] += meta_loss
                assert meta_loss.requires_grad, "meta_loss has no grad path to alpha"
                (meta_loss / batch_sz).backward()

    
            # Taking the meta gradient step (will update the learning rates)
            # self.zero_grads()

            # meta_loss = sum(meta_losses)/len(meta_losses)

            # meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)

            if self.args.learn_lr:
                self.opt_lr.step()

            if self.args.sync_update:
                self.opt_wt.step()
            else:
                with torch.no_grad():
                    for i, p in enumerate(self.net.parameters()):
                        g = p.grad
                        if g is None:
                            continue                       # <-- skip params without grads
                        lr_i = torch.relu(self.net.alpha_lr[i])
                        p.add_(-lr_i * g)                  # inplace, safe under no_grad

            # better zeroing (lower mem)
            self.net.zero_grad(set_to_none=True)
            self.net.alpha_lr.zero_grad(set_to_none=True)

        return meta_loss.item()