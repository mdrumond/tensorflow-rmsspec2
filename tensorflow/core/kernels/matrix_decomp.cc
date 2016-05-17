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
#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/platform/types.h"


#define REGISTER_MATDECOMP_OP(OpName, OpClass, Scalar)       \
  REGISTER_KERNEL_BUILDER(                                   \
      Name(OpName).Device(DEVICE_CPU).TypeConstraint<Scalar>("T"), OpClass)

namespace tensorflow{


template <typename T>
class MatrixDecompSvdBase : public OpKernel {
public:
  explicit MatrixDecompSvdBase(OpKernelConstruction* context) : OpKernel(context) {}

  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  
private:

  virtual void ComputeSVD(OpKernelContext* context,
                          const ConstMatrixMap& input,
                          MatrixMap* outU, MatrixMap* outS, MatrixMap* outV)  = 0;

  virtual int64 GetSmallDim(OpKernelContext* context,
                           const TensorShape& input_matrix_shape) = 0;
  
  void Compute(OpKernelContext* context) override{
    const Tensor& in = context->input(0);
    const TensorShape input_matrix_shape = TensorShape({in.dim_size(0),
          in.dim_size(1)});

    const int input_rank = in.dims();
    OP_REQUIRES(context, input_rank == 2,
                errors::InvalidArgument("Input tensor must have rank == 2"));

    // Create the matrix type soe we can operate on it
    ConstMatrixMap inMat(in.flat<T>().data(),
                         input_matrix_shape.dim_size(0),
                         input_matrix_shape.dim_size(1));


    const int64 inSmallDim = GetSmallDim(context, input_matrix_shape);
    // Allocate output
    // U
    Tensor* outU = nullptr;
    TensorShape outU_shape( { input_matrix_shape.dim_size(0) , inSmallDim } );
    OP_REQUIRES_OK(context, context->allocate_output(0, outU_shape, &outU));
    MatrixMap outUMat(outU->flat<T>().data(),
                      outU_shape.dim_size(0), outU_shape.dim_size(1));
    
    // S
    Tensor* outS = nullptr;
    TensorShape outS_shape( { inSmallDim } );
    OP_REQUIRES_OK(context, context->allocate_output(1, outS_shape, &outS));
    MatrixMap outSMat(outS->flat<T>().data(),
                      outS_shape.dim_size(0), 1);

    // V
    Tensor* outV = nullptr;
    TensorShape outV_shape( { input_matrix_shape.dim_size(1) , inSmallDim  } );
    OP_REQUIRES_OK(context, context->allocate_output(2, outV_shape, &outV));
    MatrixMap outVMat(outV->flat<T>().data(),
                      outV_shape.dim_size(0), outV_shape.dim_size(1));

    // Do the magic
    ComputeSVD(context, inMat, &outUMat, &outSMat, &outVMat);
  }
};

  
template <typename T>
class MatrixDecompSvd : public MatrixDecompSvdBase<T> {
public:
  explicit MatrixDecompSvd(OpKernelConstruction* context)
    : MatrixDecompSvdBase<T>(context) {}
  
  int64 GetSmallDim(OpKernelContext* context,
                   const TensorShape& input_matrix_shape) override {
    return ((input_matrix_shape.dim_size(0) < input_matrix_shape.dim_size(1))?
            input_matrix_shape.dim_size(0) : input_matrix_shape.dim_size(1));
  }

  using typename MatrixDecompSvdBase<T>::Matrix;
  using typename MatrixDecompSvdBase<T>::ConstMatrixMap;
  using typename MatrixDecompSvdBase<T>::MatrixMap;
  
  void ComputeSVD(OpKernelContext* context,
                  const ConstMatrixMap& input,
                  MatrixMap* outU, MatrixMap* outS, MatrixMap* outV) override {

    if (input.rows() == 0 || input.rows() == 0) {
      // The result is the empty matrix.
      return;
    }
    auto inJacobi = input.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    *outU = inJacobi.matrixU();
    *outS = inJacobi.singularValues();
    *outV = inJacobi.matrixV();    
  }
};
  
REGISTER_MATDECOMP_OP("MatrixDecompSvd", (MatrixDecompSvd<float>),
                          float);
REGISTER_MATDECOMP_OP("MatrixDecompSvd", (MatrixDecompSvd<double>),
                          double);

template <typename T>
class MatrixDecompSvdRand : public MatrixDecompSvdBase<T> {
public:
  explicit MatrixDecompSvdRand(OpKernelConstruction* context)
    : MatrixDecompSvdBase<T>(context) {}

  int64 GetSmallDim(OpKernelContext* context,
                    const TensorShape& input_matrix_shape) override {
    return context->input(1).flat<int32>()(0);
  }

  using typename MatrixDecompSvdBase<T>::Matrix;
  using typename MatrixDecompSvdBase<T>::ConstMatrixMap;
  using typename MatrixDecompSvdBase<T>::MatrixMap;
  
  void ComputeSVD(OpKernelContext* context,
                  const ConstMatrixMap& input,
                  MatrixMap* outU, MatrixMap* outS, MatrixMap* outV) override {

    if (input.rows() == 0 || input.cols() == 0) {
      // The result is the empty matrix.
      return;
    }

    int64 svdSmallDim = GetSmallDim(context,
                                    TensorShape( {input.rows(), input.cols()} ));
    int64 inSmallDim = (input.rows() > input.cols())? input.cols() : input.rows();
    int64 inBigDim = (input.rows() > input.cols())? input.rows() : input.cols();
    bool transpose = (input.cols() > input.rows())? true : false;

    Matrix inMat = transpose? input.transpose().eval() : input;
    if(transpose){
      inMat = input.transpose();
    }
    else {
      inMat = input;
    }

    Matrix randIn(Matrix::Random(inSmallDim, svdSmallDim));
    Matrix id(Matrix::Identity(inBigDim, svdSmallDim));
    auto q = (inMat * randIn).householderQr().householderQ()*id;
    
    auto qInJacobi = (q.transpose() * inMat).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    auto uHat = qInJacobi.matrixU();
    auto s = qInJacobi.singularValues();
    auto v = qInJacobi.matrixV();

    auto u = q*uHat;

    if(transpose){
      *outU = v;
      *outS = s;
      *outV = u;
    }
    else {
      *outU = u;
      *outS = s;
      *outV = v;
    }
  }
};

  
REGISTER_MATDECOMP_OP("MatrixDecompSvdRand", (MatrixDecompSvdRand<float>),
                          float);
REGISTER_MATDECOMP_OP("MatrixDecompSvdRand", (MatrixDecompSvdRand<double>),
                          double);
}  // namespace tensorflow
