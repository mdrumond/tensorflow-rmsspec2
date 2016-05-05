#! /bin/bash

if [ -z "$1" ]; then
    LOG_DIR=logs
else
    LOG_DIR=$1
fi

python3 /home/parsa/tensorflow/tensorflow/tensorboard/tensorboard.py --logdir $LOG_DIR --host 0.0.0.0 
