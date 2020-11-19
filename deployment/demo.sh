#!/usr/bin/env bash
set -e

source venv/bin/activate

./generate_raw_data.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4
./convert_model.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4

echo Building TFLITE_X86_64 target
(cd src/tflite_x86_64 && ./build.sh -p ../../data -w 30225)

echo Starting TFLITE_X86_64
(cd src/tflite_x86_64 && ./pipeline)

echo Demo complete