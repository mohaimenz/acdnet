#!/usr/bin/env bash

set -e

current_dir=`pwd`

if [ -d $current_dir/spresense/examples ]
then
    echo Adding symlink to tflite_micro project folder
    ln -s $current_dir/../src/tflite_micro  $current_dir/spresense/examples/tflite_micro
else
    echo Spresense examples folder not found. Check spresense installation
    exit 1
fi

if [ -d $current_dir/spresense/sdk/configs ]
then
    echo Adding symlink to tflite_micro_config project folder
    ln -s $current_dir/../src/tflite_micro_config  $current_dir/spresense/sdk/configs/examples/tflite_micro
else 
    echo Spresense sdk config folder not found. Check spresense installation
    exit 1
fi

if [ -d $current_dir/tensorflow ]
then
    
    if [ -d $current_dir/spresense/examples/tflite_micro/tensorflow ]
    then
        echo Adding symlink to TensorFlow in tflite_micro project folder
        ln -s $current_dir/tensorflow  $current_dir/spresense/examples/tflite_micro/tensorflow
    else 
        echo Skipping install of Tensorflow symlink 
    fi

else
    echo Tensorflow folder not found. Check spresense installation
    exit 1
fi
