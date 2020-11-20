#!/usr/bin/env bash
current_dir=`pwd`

cd ../../ext/spresense/sdk 
source ~/spresenseenv/setup 

make distclean

./tools/config.py examples/tflite_micro
make

cd $current_dir
echo Build task finished
