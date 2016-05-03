/* Copyright 2015 Google Inc. All Rights Reserved.

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

// SVD decompositions

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SVD"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow{

template <class Scalar, bool SupportsBatchOperationT>
class MatrixDecompSvdSOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixDecompSvdSOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) { }

  ~MatrixDecompSvdSOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    const int64 smallDim = (input_matrix_shape.dim_size(0) < input_matrix_shape.dim_size(1))?
      input_matrix_shape.dim_size(0) : input_matrix_shape.dim_size(1);
    return TensorShape({ smallDim });
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 cols = input_matrix_shape.dim_size(1);
    // TODO: figure out the actual cost
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return 2 * rows * cols;
    }
  }
  
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::Matrix
          Matrix;
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap
          MatrixMap;
  typedef typename UnaryLinearAlgebraOp<
      Scalar, SupportsBatchOperationT>::ConstMatrixMap ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& matrix,
                     MatrixMap* output) override {
    const int64 rows = matrix.rows();
    const int64 cols = matrix.cols();

    if (rows == 0 || cols == 0) {
      // The result is the empty matrix.
      return;
    }
    
    *output = matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).singularValues();
  }

};

REGISTER_LINALG_OP("MatrixDecompSvdS", (MatrixDecompSvdSOp<float, false>),
                          float);
REGISTER_LINALG_OP("MatrixDecompSvdS", (MatrixDecompSvdSOp<double, false>),
                          double);

  template <class Scalar, bool SupportsBatchOperationT>
class MatrixDecompSvdUOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixDecompSvdUOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) { }

  ~MatrixDecompSvdUOp() override {}

  TensorShape GetOutputMatrixShape(
                                   const TensorShape& input_matrix_shape) override {
    const int64 smallDim = (input_matrix_shape.dim_size(0) < input_matrix_shape.dim_size(1))?
      input_matrix_shape.dim_size(0) : input_matrix_shape.dim_size(1);
    
    return TensorShape({ input_matrix_shape.dim_size(0),
          smallDim});
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 cols = input_matrix_shape.dim_size(1);
    // TODO: figure out the actual cost
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return 2 * rows * cols;
    }
  }
  
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::Matrix
          Matrix;
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap
          MatrixMap;
  typedef typename UnaryLinearAlgebraOp<
      Scalar, SupportsBatchOperationT>::ConstMatrixMap ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& matrix,
                     MatrixMap* output) override {
    const int64 rows = matrix.rows();
    const int64 cols = matrix.cols();

    if (rows == 0 || cols == 0) {
      // The result is the empty matrix.
      return;
    }
    
    *output = matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).matrixU();
  }

};

REGISTER_LINALG_OP("MatrixDecompSvdU", (MatrixDecompSvdUOp<float, false>),
                          float);
REGISTER_LINALG_OP("MatrixDecompSvdU", (MatrixDecompSvdUOp<double, false>),
                          double);

  template <class Scalar, bool SupportsBatchOperationT>
class MatrixDecompSvdVOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixDecompSvdVOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) { }

  ~MatrixDecompSvdVOp() override {}

  TensorShape GetOutputMatrixShape(
                                   const TensorShape& input_matrix_shape) override {
    const int64 smallDim = (input_matrix_shape.dim_size(0) < input_matrix_shape.dim_size(1))?
      input_matrix_shape.dim_size(0) : input_matrix_shape.dim_size(1);
    
    return TensorShape({ input_matrix_shape.dim_size(1), smallDim });
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 cols = input_matrix_shape.dim_size(1);
    // TODO: figure out the actual cost
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return 2 * rows * cols;
    }
  }
  
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::Matrix
          Matrix;
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap
          MatrixMap;
  typedef typename UnaryLinearAlgebraOp<
      Scalar, SupportsBatchOperationT>::ConstMatrixMap ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& matrix,
                     MatrixMap* output) override {
    const int64 rows = matrix.rows();
    const int64 cols = matrix.cols();

    if (rows == 0 || cols == 0) {
      // The result is the empty matrix.
      return;
    }
    
    *output = matrix.jacobiSvd(Eigen::ComputeThinV | Eigen::ComputeThinU).matrixV();
  }

};

REGISTER_LINALG_OP("MatrixDecompSvdV", (MatrixDecompSvdVOp<float, false>),
                          float);
REGISTER_LINALG_OP("MatrixDecompSvdV", (MatrixDecompSvdVOp<double, false>),
                          double);


    template <class Scalar, bool SupportsBatchOperationT>
class MatrixDecompQrQOp
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixDecompQrQOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) { }

  ~MatrixDecompQrQOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return TensorShape({ input_matrix_shape.dim_size(0),
          input_matrix_shape.dim_size(0)});
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    const int64 cols = input_matrix_shape.dim_size(1);
    // TODO: figure out the actual cost
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
      return kint32max;
    } else {
      return 2 * rows * cols;
      ;    }
  }
  
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::Matrix
          Matrix;
  typedef
      typename UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap
          MatrixMap;
  typedef typename UnaryLinearAlgebraOp<
      Scalar, SupportsBatchOperationT>::ConstMatrixMap ConstMatrixMap;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& matrix,
                     MatrixMap* output) override {
    const int64 rows = matrix.rows();
    const int64 cols = matrix.cols();
    if (rows == 0 || cols == 0) {
      // The result is the empty matrix.
      return;
    }
    
    *output = matrix.householderQr().householderQ();
  }

};

REGISTER_LINALG_OP("MatrixDecompQrQ", (MatrixDecompQrQOp<float, false>),
                          float);
REGISTER_LINALG_OP("MatrixDecompQrQ", (MatrixDecompQrQOp<double, false>),
                          double);

}  // namespace tensorflow
