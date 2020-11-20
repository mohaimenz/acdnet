#!/usr/bin/env bash

set -e

exit_on_error() {
    exit_code=$1
    last_command=${@:2}
    if [ $exit_code -ne 0 ]; then
        >&2 echo "\"${last_command}\" command failed with exit code ${exit_code}."
        exit $exit_code
    fi
}

echo 'Setup - Installing python virtual environment'

if [ -d venv ]
then
    echo Python virtual environment already exists. Skipping.
else
    python3 -m venv venv
fi

echo 'Setup - Installing python requirements'
source venv/bin/activate && pip install -r requirements.txt
exit_on_error $? !!

echo 'Tensorflow - installing tensorflow'
(cd ext && ./install_tensorflow.sh)
exit_on_error $? !!

echo 'Spresense SDK - Installing Spresense SDK'
(cd ext && ./install_spresense_sdk.sh)
exit_on_error $? !!

echo 'Spresense SDK - Installing symlinks'
(cd ext && ./install_symlinks.sh)
exit_on_error $? !!

echo 'Flatbuffer - Installing flatbuffer'
(cd ext && ./install_flatbuffers.sh)
exit_on_error $? !!
