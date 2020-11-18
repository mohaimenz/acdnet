#!/usr/bin/env bash

echo 'Setup - installing python virtual environment'
python3 -m venv venv

echo 'Setup - installing python requirements'
source venv/bin/activate && pip install -r requirements.txt

echo 'Nuttx - installing libcxx'
(cd ext && ./install_libcxx.sh)

echo 'Nuttx - installing symlinks'
(cd ext && ./install_symlinks.sh)

echo 'TensorFlow - Downloading TFLite dependencies'
(cd ext/tensorflow && ./tensorflow/lite/tools/make/download_dependencies.sh)

echo 'TensorFlow - Building TensorFlow Lite'
(cd ext/tensorflow && ./tensorflow/lite/tools/make/build_lib.sh)

echo 'TensorFlow - Building TensorFlow Lite Micro'
(cd ext/tensorflow && make -f tensorflow/lite/micro/tools/make/Makefile)