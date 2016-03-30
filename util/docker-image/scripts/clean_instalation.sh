#! /bin/bash

bazel clean
pip uninstall tensorflow
rm -rf _python_build
