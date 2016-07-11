#! /bin/bash
read -r -d '' SGD_FLAGS << EOM
--training-method=sgd \
--batch-size=128 \
--learning-rate=0.1 \
--learning-rate-decay=0.1
EOM

read -r -d '' RMSPROP_FLAGS << EOM
--training-method=rmsprop \
--batch-size=128 \
--momentum=0.9 \
--epsilon=1E-8 \
--rms-decay=0.9 \
--learning-rate=0.1 \
--learning-rate-decay=0.1
EOM

read -r -d '' RMSSPECTRAL_FLAGS << EOM
--training-method=ssd \
--batch-size=1024 \
--momentum=0.9 \
--epsilon=1E-8 \
--rms-decay=0.9 \
--learning-rate=000.1 \
--learning-rate-decay=0.5
EOM

read -r -d '' RMSSPECTRAL_NO_MOM_FLAGS << EOM
--training-method \
--batch-size=1024 \
--momentum=0 \
--epsilon=1E-8 \
--rms-decay=0.9 \
--learning-rate=000.1 \
--learning-rate-decay=0.5
EOM

declare -A flag_dic=( ["ssd"]="$SGD_FLAGS" \
                             ["rmsprop"]="$RMSPROP_FLAGS" \
                             ["rmsspectral"]="$RMSSPECTRAL_FLAGS" \
                             ["rmsspectral-no-mom"]="$RMSSPECTRAL_NO_MOM_FLAGS")
for i in "$@"
do
case $i in
    -e=*|--experiment=*)
    EXPERIMENT="${i#*=}"
    shift # past argument=value
    ;;
    -f=*|--experiment-folder=*)
    EXPERIMENT_FOLDER="${i#*=}"
    shift # past argument=value
    ;;
    -s=*|--steps=*)
    STEPS="${i#*=}"
    shift # past argument=value
    ;;
    *)
            # unknown option
    ;;
esac

if [-z "$EXPERIMENT_FOLDER" ]; then
    EXPERIMENT_FOLDER="./results"
fi

if [-z "$STEPS" ]; then
    STEPS=10000
fi

function run_training {
    $METHOD=$1
    $TRAIN_FLAGS = flag_dic[$METHOD]
    $MAX_STEPS = $2
    $EXPERIMENT_FOLDER = $3/$METHOD
    python3 cifar10_train --max-steps=$MAX_STEPS $TRAIN_FLAGS
    mkdir -p $EXPERIMENT_FOLDER
    echo $TRAIN_FLAGS > $EXPERIMENT_FOLDER/experiment.txt
    cp -rf cifar10_train/* $EXPERIMENT_FOLDER/
}

EXPERIMENTS=()
if [ -z $EXPERIMENT ]; then
    EXPERIMENTS += ( "sgd rmsprop rmsspectral rmsspectra-no-mom" )
else
    EXPERIMENTS +=($EXPERIMENT)
fi

for i in "${EXPERIMENTS[@]}"
do
    run_training $i $STEPS $EXPERIMENT_FOLDER
done

