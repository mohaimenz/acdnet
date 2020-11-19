#!/usr/bin/env bash

set -e

CURRENT_DIR=`pwd`

TF_PATH='tensorflow'
TF_BRANCH='v2.3.1'
TF_PATCHES='patches/tensorflow'

echo 'Tensorflow - Installation commencing'

if [ -d ${TF_PATH} ] 
then 
    
    echo "Directory $TF_PATH already exists. Delete folder to reaquire."

else

    echo 'TensorFlow - Cloning repo'
    git clone --branch=$TF_BRANCH --depth=1 --recurse-submodules https://github.com/tensorflow/tensorflow.git $TF_PATH

    echo 'TensorFlow - Downloading TFLite dependencies'
    (cd $TF_PATH && ./tensorflow/lite/tools/make/download_dependencies.sh)

    echo 'TensorFlow - Apply patches'
    for f in $TF_PATCHES/*.patch
    do 
        echo $f
        [ -f "$f" ] || break
        ( cd $TF_PATH && git apply --reject --whitespace=fix $CURRENT_DIR/$f )
    done;

    
fi

echo 'TensorFlow - Building TensorFlow Lite'
(source ../venv/bin/activate && cd $TF_PATH && ./tensorflow/lite/tools/make/build_lib.sh)

echo 'TensorFlow - Building TensorFlow Lite Micro for x86_64'
(source ../venv/bin/activate && cd $TF_PATH && make -f tensorflow/lite/micro/tools/make/Makefile microlite)

echo 'TensorFlow - Building CMSIS-NN library dependency'
(source ../venv/bin/activate && cd $TF_PATH && make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn third_party_downloads)

echo 'Tensorflow - Installation complete'./