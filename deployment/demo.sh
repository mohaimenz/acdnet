#!/usr/bin/env bash
set -e

source venv/bin/activate

./generate_raw_data.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4
./convert_model.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4

echo Building TFLITE_X86_64 target
(cd src/tflite_x86_64 && ./build.sh -p ../../data -w 30225)

echo Starting TFLITE_X86_64
(cd src/tflite_x86_64 && ./pipeline)

read -n 1 -s -r -p "Press any key to continue"

echo Building TFLITE_MICRO target
(cd src/tflite_micro && ./build.sh)

echo 1. Copy the 'deployment/data' folder to your microSD card before starting TFLITE_MICRO application
echo 2. Place microSD card in Spresense Extension board and connect with USB to Spresense Main board
echo 3. Type `upload.sh` to transfer TFLITE_MICRO to Spresense
echo 4. In Nuttx serial console, type `tflite_micro` to start

echo Demo complete