#!/usr/bin/env bash
set -e

source venv/bin/activate
./generate_raw_data.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4
./convert_model.py ../tf/resources/pretrained_models/acdnet20_20khz_fold4.h5 --fold 4
