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

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import training


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
                 use_locking=False,
                 name="RMSSpectral"):
        super(RMSSpectralOptimizer, self).__init__(learning_rate, decay,
                                                   momentum, epsilon,
                                                   use_locking, name)

    def _sharpOp(self, vector):
        u = linalg_ops.matrix_decomp_svd_u(vector)
        s = linalg_ops.matrix_decomp_svd_s(vector)
        v = linalg_ops.matrix_decomp_svd_v(vector)

        return (math_ops.matmul(u, array_ops.transpose(v)) *
                math_ops.reduce_sum(s))

    def _apply_rms_spectral(self, grad, var):
        # see if variable updates need something special
        # might have to resize the variables (they are suposedly flat)
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")

        momentum = math_ops.cast(self._momentum_tensor, var.dtype.base_dtype)
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        decay = math_ops.cast(self._decay_tensor, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype)

        rms_update = rms.assign(decay * rms +
                                (1 - decay) *
                                math_ops.square(grad))
        aux = math_ops.sqrt(math_ops.sqrt(rms_update)+epsilon)
        update = (lr *
                  (self._sharpOp(grad / aux) /
                   aux))

        mom_update = mom.assign(mom * momentum + update)
        var_update = var.assign_sub(mom_update)

        return control_flow_ops.group(*[var_update, rms_update, mom_update])

    def _apply_dense(self, grad, var):
        # what about bias???
        # check var shapes (might not work if this thing is flat):
        # 2 dims - rms spectral var.shape!!

        # 4 or 1 dims - rms prop (convolutional layers, bias)
        print(grad.get_shape())
        if ((len(grad.get_shape()) == 1) or (len(grad.get_shape()) == 4)):
            print("applying rms prop")
            return super()._apply_dense(grad, var)
        else:
            print("applying rms spectral")
            return self._apply_rms_spectral(grad, var)
