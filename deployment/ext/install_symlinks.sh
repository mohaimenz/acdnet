#!/usr/bin/env bash

set -e

current_dir=`pwd`

TFLITE_MICRO_CHECK=$current_dir/spresense/examples
TFLITE_MICRO_ALIAS=$current_dir/spresense/examples/tflite_micro
TFLITE_MICRO_TARGET=$current_dir/../src/tflite_micro

TFLITE_MICRO_CFG_CHECK=$current_dir/spresense/sdk/configs
TFLITE_MICRO_CFG_ALIAS=$current_dir/spresense/sdk/configs/examples/tflite_micro
TFLITE_MICRO_CFG_TARGET=$current_dir/../src/tflite_micro_config

TFLITE_MICRO_TF_CHECK=$current_dir/tensorflow
TFLITE_MICRO_TF_ALIAS=$current_dir/spresense/examples/tflite_micro/tensorflow
TFLITE_MICRO_TF_TARGET=$current_dir/tensorflow

if [ ! -L $TFLITE_MICRO_ALIAS ]
then

    if [ -d $TFLITE_MICRO_CHECK ]
    then
        echo Adding symlink to tflite_micro project folder
        ln -s $TFLITE_MICRO_TARGET $TFLITE_MICRO_ALIAS
    else
        echo Spresense examples folder $TFLITE_MICRO_CHECK not found. Check spresense installation
        exit 1
    fi

else
    echo Spresense example $TFLITE_MICRO_ALIAS already exists. Skipping.
fi

if [ ! -L $TFLITE_MICRO_CFG_ALIAS ]
then

    if [ -d $TFLITE_MICRO_CFG_CHECK ]
    then
        echo Adding symlink to tflite_micro_config project folder
        ln -s $TFLITE_MICRO_CFG_TARGET  $TFLITE_MICRO_CFG_ALIAS
    else 
        echo Spresense sdk config folder $TFLITE_MICRO_CFG_CHECK not found. Check spresense installation
        exit 1
    fi
else
    echo Spresense config $TFLITE_MICRO_CFG_ALIAS already exists. Skipping.
fi

if [ ! -L $TFLITE_MICRO_TF_ALIAS ]
then

    if [ -d $TFLITE_MICRO_TF_CHECK ]
    then
        
        echo Adding symlink to TensorFlow within tflite_micro project folder
        ln -s $TFLITE_MICRO_TF_TARGET $TFLITE_MICRO_TF_ALIAS

    else
        echo Tensorflow folder $TFLITE_MICRO_TF_CHECK not found. Check spresense installation
        exit 1
    fi
else
    echo TensorFlow link $TFLITE_MICRO_TF_ALIAS already exists. Skipping.
fi