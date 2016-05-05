#! /bin/bash

if [ "$1" -eq "r" ]; then
    bazel clean 
fi

pip uninstall tensorflow
rm -rf _python_build

PYTHON_BIN_PATH=/usr/bin/python3 TF_NEED_CUDA=0 ./configure
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package --spawn_strategy=standalone --genrule_strategy=standalone --jobs 4

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
