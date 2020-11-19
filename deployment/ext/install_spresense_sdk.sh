#!/usr/bin/env bash

CURRENT_DIR=`pwd`

SPRESENSE_PATH='spresense'
SPRESENSE_URL='https://github.com/sonydevworld/spresense.git'
SPRESENSE_BRANCH='v2.0.1'

LIBCXX_URL='https://bitbucket.org/acassis/libcxx'
LIBCXX_PATH='spresense/nuttx'

if id -nG "$USER" | grep -qw "$GROUP"; then
    echo $USER belongs to $GROUP
else
    echo Adding user to the dialout group
    sudo usermod -a -G dialout \$USER
    echo Log out, to join group
fi

if [ -d ${SPRESENSE_PATH} ] 
then
    echo Directory $SPRESENSE_PATH already exists. Delete path to reacquire
else
    git clone --branch=$SPRESENSE_BRANCH --depth=1 --recurse-submodules $SPRESENSE_URL $SPRESENSE_PATH
fi

if [-d ${LIBCXX_PATH}/libs/libxx/libcxx ]
then
    echo Skipping the install of libcxx, as path already exists
else
    
    git clone $LIBCXX_URL --depth=1
    (cd libcxx; ./install.sh $CURRENT_DIR/spresense/nuttx)
    cp patches/libcxx/optional.cxx $CURRENT_DIR/spresense/nuttx/libs/libxx/libcxx
    rm -rf libcxx
fi