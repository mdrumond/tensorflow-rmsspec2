"""

Read a checkpoints and export its weights to a numpy variable

"""

import os
import sys

import numpy as np
import tensorflow as tf

def getTensor(file_name):
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        print(reader.debug_string().decode("utf-8"))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))



if __name__ == "__main__":
    getTensor(sys.argv[1])
    
