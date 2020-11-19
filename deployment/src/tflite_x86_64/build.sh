#!/usr/bin/env bash

while getopts p:w: flag
do
    case "${flag}" in
        p) FEATURE_PATH=${OPTARG};;
        w) FEATURE_WIDTH=${OPTARG};;
    esac
done

echo $0 -p [feature_path] -w [width]
echo Feature Path:  $FEATURE_PATH
echo Feature Width: $FEATURE_WIDTH

if [ ! -d $FEATURE_PATH ]
then 
    echo Feature data path is required
fi

function config {
    aclocal
    automake --add-missing
    autoconf
    ./configure
}

config
make distclean

config
make FEATURE_PATH=$FEATURE_PATH FEATURE_WIDTH=$FEATURE_WIDTH
