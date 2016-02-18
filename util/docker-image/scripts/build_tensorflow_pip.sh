
PYTHON_BIN_PATH=/usr/bin/python TF_NEED_CUDA=0 ./configure
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow-0.6.0-py2-none-any.whl
