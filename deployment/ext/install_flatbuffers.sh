#!/usr/bin/env bash
FB_URL=https://github.com/google/flatbuffers.git
FB_PATH='flatbuffers'
FB_BRANCH='v1.12.0'

if [ -d $FB_PATH ]
then
    echo Flatbuffers already installed. Skipping.
else
    git clone --depth=1 --branch $FB_BRANCH $FB_URL $FB_PATH
fi
