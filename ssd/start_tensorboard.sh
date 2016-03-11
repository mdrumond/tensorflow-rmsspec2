#! /bin/bash

python3 ../tensorflow/tensorboard/tensorboard.py --logdir logs --host 0.0.0.0 &> /dev/null &
