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

"""One-line documentation for rmsprop module.

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

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class RMSSpectralOptimizer(optimizer.Optimizer):
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
        """Construct a new RMSSpectral optimizer.

        Args:
        learning_rate: A Tensor or a floating point value.  The learning rate.
        decay: Discounting factor for the history/coming gradient
        momentum: A scalar tensor.
        epsilon: Small value to avoid zero denominator.
        use_locking: If True use locks for update operation.
        name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSSpectral".
        """
        super(RMSSpectralOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        
        # Tensors for learning rate and momentum.  Created in _prepare.
        self._learning_rate_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None
        
    def _sharpOp(self,vector):
        u = tf.matrix_decomp_svd_u(vector)
        s = tf.matrix_decomp_svd_s(vector)
        v = tf.matrix_decomp_svd_v(vector)

        return tf.matmul(u,tf.transpose(v))*tf.reduce_sum(s)

    def _apply_rms_spectral(self,var,rms,mom,learning_rate,
                            decay,momentum,epislon,alpha):
        # see if variable updates need something special
        # might have to resize the variables (they are suposedly flat)
        rms.assign(alpha * rms + (1 - alpha) * grad)
        aux = ops.sqrt(ops.sqrt(rms)+epsilon)
        var.assign_sub(-learning_rate * ops.div(_sharpOp(ops.div(grad,aux)),aux))
      
    def _create_slots(self, var_list):
        for v in var_list:
            val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, val, "rms", self._name)
            self._zeros_slot(v, "momentum", self._name)

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                           name="learning_rate")
        self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
        self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                                      name="momentum")
        self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                                     name="epsilon")

    def _apply_dense(self, grad, var):
        # what about bias???
        # check var shapes (might not work if this thing is flat):
        # 2 dims - rms spectral var.shape!!
        # 4 dims - rms prop (convolutional layers)
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        return _apply_rms_spectral(
            var, rms, mom,
            self._learning_rate_tensor,
            self._decay_tensor,
            self._momentum_tensor,
            self._epsilon_tensor,
            grad, use_locking=self._use_locking).op
    
    def _apply_sparse(self, grad, var):
        raise NotImplementedError()
