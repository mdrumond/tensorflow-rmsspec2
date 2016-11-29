#! /bin/bash

if [ "$1" = "r" ]; then
    bazel clean 
    pip3 uninstall tensorflow
    rm -rf _python_build
    PYTHON_BIN_PATH=/usr/bin/python3 TF_NEED_CUDA=1 ./configure
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
