# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""One-line documentation for rmsspectral module.

rmsprop algorithm [tieleman2012rmsprop]

A detailed description of rmsprop.

- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average

mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t / sqrt(mean_square + epsilon)
delta = - mom

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import training
from tensorflow.python.training import training_ops


class RMSSpectralOptimizer(training.rmsprop.RMSPropOptimizer):
    """Optimizer that implements the RMSSpectral algorithm.

    See the [paper]
    (http://papers.nips.cc/paper/5795-preconditioned-spectral-descent-for-deep-learning.pdf).

    @@__init__
    """

    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 svd_approx_size=30,
                 use_approx_sharp=True,
                 use_locking=False,
                 name="RMSSpectral"):
        super(RMSSpectralOptimizer, self).__init__(learning_rate, decay,
                                                   momentum, epsilon,
                                                   use_locking, name)
        self._svd_approx_size = svd_approx_size
        self._use_approx_sharp = use_approx_sharp

    def _create_slots(self, var_list):
        for v in var_list:
            if ((len(v.get_shape()) == 1) or (len(v.get_shape()) == 4)):
                val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
                val1 = constant_op.constant(0.0, dtype=v.dtype, shape=v.get_shape())
            else:
                val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())          
                val1 = constant_op.constant(0.0, dtype=v.dtype, shape=v.get_shape())          
            self._get_or_make_slot(v, val, "rms", self._name)
            self._get_or_make_slot(v, val1, "momentum", self._name)
        
    def _apply_rms_spectral(self, grad, var):
        # see if variable updates need something special
        # might have to resize the variables (they are suposedly flat)
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")

        return training_ops.apply_rms_spectral(
                    var, rms, mom,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
            math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
            grad, use_locking=self._use_locking,
            use_approx_sharp=self._use_approx_sharp).op

    def _apply_dense(self, grad, var):
        # what about bias???
        # check var shapes (might not work if this thing is flat):
        # 2 dims - rms spectral var.shape!!

        # 4 or 1 dims - rms prop (convolutional layers, bias)
        if ((len(grad.get_shape()) == 1) or (len(grad.get_shape()) == 4)):
            return super()._apply_dense(grad, var)
        else:
            return self._apply_rms_spectral(grad, var)
