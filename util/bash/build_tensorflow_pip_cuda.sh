#! /bin/bash

if [ "$1" = "r" ]; then
    bazel clean 
    pip3 uninstall tensorflow
    rm -rf _python_build
    PYTHON_BIN_PATH=/usr/bin/python3 \
                   TF_NEED_CUDA=1 \
                   TF_NEED_GCP=0 \
                   TF_NEED_HDFS=0 \
                   GCC_HOST_COMPILER_PATH=$(which gcc) \
                   CUDA_TOOLKIT_PATH=/usr/local/cuda \
                   TF_CUDA_VERSION=8.0 \
                   CUDNN_INSTALL_PATH=/usr/local/cuda \
                   TF_CUDNN_VERSION=5 \
                   TF_CUDA_COMPUTE_CAPABILITIES="3.5,5.2" \
                   ./configure
fi


bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

if [ $? -eq 0 ]; then
    echo "### Succesiful build, installing pip package"
    mkdir _python_build
    cd _python_build
    ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
    ln -s ../tensorflow/tools/pip_package/* .
    python3 setup.py develop --user
else
    echo "### Build failed"
fi
