/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/contrib/stochastic_quantization/stocquant_matmul_op.h"

// For now use 32 bit ints as a representation. Figure out other things later.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class StocquantMatMulOp : public OpKernel {
public:
  explicit QuantizedMatMulOp(OpKernelConstruction* context)
    : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
  }
}

} // namespace tensorflow
