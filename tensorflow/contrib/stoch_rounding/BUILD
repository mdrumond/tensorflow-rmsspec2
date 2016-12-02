# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_custom_op_library",
)

cc_library(
    name = "cc_round_ops",
    srcs = ["ops/round_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cc_ops",
    linkstatic = 1,
    deps = [
        ":cc_round_ops",
    ],
    alwayslink = 1,
)

py_library(
    name = "stoch_rounding_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow/contrib/stoch_rounding:stoch_rounding_ops_py",
        "//tensorflow/contrib/stoch_rounding/kernels:stoch_rounding_kernels_py",
    ],
)

py_library(
    name = "ops",
    srcs = [
        "python/round_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":round_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "round_ops",
    deps = ["//tensorflow/contrib/quantization:cc_round_ops"],
)

py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["python/dequantize_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/quantization:quantized_ops_py",
        "//tensorflow/contrib/quantization/kernels:quantized_kernels_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["python/quantized_conv_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/quantization:quantized_ops_py",
        "//tensorflow/contrib/quantization/kernels:quantized_kernels_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

filegroup(
    name = "py_srcs",
    data = glob([
        "**/*.py",
    ]),
)

tf_custom_op_library(
    name = "_stoch_rounding_ops.so",
    srcs = [
        "ops/round_ops.cc",
    ],
    deps = [
    ],
)

py_library(
    name = "stoch_rounding_ops_py",
    srcs = ["load_stoch_rounding_ops_so.py"],
    data = ["_stoch_rounding_ops.so"],
    srcs_version = "PY2AND3",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
