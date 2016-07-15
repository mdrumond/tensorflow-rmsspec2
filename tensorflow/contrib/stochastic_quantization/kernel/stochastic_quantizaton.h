/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_STOCHASTIC_QUANTIZATION_H_
#define TENSORFLOW_CONTRIB_STOCHASTIC_QUANTIZATION_H_

#include "tensorflow/core/framework/tensor.h"

// Stochastic rounding for floats, int32, etc...
namespace tensorflow {

// {float, int32, ... to fixed point N_BITS_INT.N_BITS_FRAC
template <class T, int N_BITS_INT, int N_BITS_FRAC>
void roundStochastic(const Tensor& input, Tensor* result){
  
  auto flat_input = input.flat<T>();
  Tensor flat_rand_round(DT_INT32, flat_input.shape())
  auto flat_result = result->flat<int32>();
  DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  DCHECK_EQ(DT_INT32, result.dtype());

  // do the magic

  return result
}

// fixed point N_BITS_INT.N_BITS_FRAC to {float, int32 ..}
template <class T, int N_BITS_INT, int N_BITS_FRAC>
void StocquantToFloat(const Tensor& input, Tensor* result){
}
                      
 
 
} // namespace tensorflow
#endif // TENSORFLOW_CONTRIB_STOCHASTIC_QUANTIZATION_H_
