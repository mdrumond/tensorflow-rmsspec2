#! /bin/bash

TRAINER=$1
TRAINER_DIR="cifar10_$1"
CHECK_DIR="$TRAINER_DIR/checkpoints"
SUMMARIES_DIR="$TRAINER_DIR/summaries"


if [ -z "$1" ]; then
    echo "Error, no trainer name"
    exit 1
fi

mkdir $TRAINER_DIR
mkdir $CHECK_DIR
mkdir $SUMMARIES_DIR

mv ./cifar10_train/* $CHECK_DIR
mv /tmp/cifar10_eval/* $SUMMARIES_DIR

mv $TRAINER_DIR ~/
