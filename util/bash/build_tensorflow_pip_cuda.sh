#! /bin/bash

if [ "$1" = "r" ]; then
    bazel clean 
    pip3 uninstall tensorflow
    rm -rf _python_build
fi


PYTHON_BIN_PATH=/usr/bin/python3 \
               TF_NEED_CUDA=1 \
               TF_NEED_GCP=0 \
               GCC_HOST_COMPILER_PATH=$(which gcc) \
               CUDA_TOOLKIT_PATH=/usr/local/cuda \
               CUDNN_INSTALL_PATH=/usr/local/cuda \
               CUDNN_INSTALL_PATH=`${PYTHON_BIN_PATH} -c "import os; print(os.path.realpath(os.path.expanduser('${CUDNN_INSTALL_PATH}')))"` \
               ./configure
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --spawn_strategy=standalone --genrule_strategy=standalone --jobs 4 

if [ $? -eq 0 ]; then
    echo "### Succesiful build, installing pip package"
    mkdir _python_build
    cd _python_build
    ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .
    ln -s ../tensorflow/tools/pip_package/* .
    python3 setup.py develop --user
else
    echo "### Build failed"
fi
