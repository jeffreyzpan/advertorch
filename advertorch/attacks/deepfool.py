# Copyright (c) 2019-present, Jeffrey Pan
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This is an implementation of DeepFool based on the official implementation at https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from .base import Attack
from .base import LabelMixin
from .utils import rand_init_delta

import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

class DeepFoolAttack(Attack, LabelMixin):
    """
    The DeepFool attack, https://arxiv.org/abs/1511.04599
    
    """
    def __init__(self, predict, loss_fn=None, num_classes=10, overshoot=0.02,
            max_iter=50, clip_min=0., clip_max=1.):
        """
        Create an instance of the PGDAttack.
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for deepfool (default = 50)
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param loss_fn: loss function
        """
        super(DeepFoolAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        input_shape = x.shape
        pert_inputs = copy.deepcopy(x)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0
        y_i = y

        #TODO replace this code with a more meaningful call and understand what this actually does lol
        f_image = self.predict(x[0]).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        fs = self.predict(x)
        fs_list = [fs[0,]]

        while y_i == y and loop_i < self.max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = x + (1+overshoot)*torch.from_numpy(r_tot)

            fs = self.perdict(pert_image)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1+overshoot)*r_tot

        return r_tot, loop_i, y, k_i, pert_image