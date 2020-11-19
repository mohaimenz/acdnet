#!/usr/bin/env bash

CURRENT_DIR=`pwd`

SPRESENSE_PATH='spresense'
SPRESENSE_URL='https://github.com/sonydevworld/spresense.git'
SPRESENSE_BRANCH='v2.0.1'
SPRESENSE_PATCHES='patches/spresense'

LIBCXX_URL='https://bitbucket.org/acassis/libcxx'
LIBCXX_PATH='spresense/nuttx'

NUTTX_PATH='spresense/nuttx'
NUTTX_PATCHES='patches/nuttx'

GROUP=dialout

wget https://raw.githubusercontent.com/sonydevworld/spresense/master/install-tools.sh
chmod 775 install-tools.sh
./install-tools.sh
rm install-tools.sh

if [ -d ${SPRESENSE_PATH} ] 
then
    echo Directory $SPRESENSE_PATH already exists. Delete path to reacquire
else
    git clone --branch=$SPRESENSE_BRANCH --depth=1 --recurse-submodules $SPRESENSE_URL $SPRESENSE_PATH
fi

echo 'Spresense SDK - Apply patches'
for f in $SPRESENSE_PATCHES/*.patch
do 
    echo $f
    [ -f "$f" ] || break
    ( cd $SPRESENSE_PATH && git apply --reject --whitespace=fix $CURRENT_DIR/$f )
done;

echo 'Libcxx - Apply patches'
if [ -d ${LIBCXX_PATH}/libs/libxx/libcxx ]
then
    echo Skipping the install of libcxx, as path already exists
else
    echo Installing Libcxx

    git clone $LIBCXX_URL --depth=1
    (cd libcxx; ./install.sh $CURRENT_DIR/spresense/nuttx)
    cp patches/libcxx/optional.cxx $CURRENT_DIR/spresense/nuttx/libs/libxx/libcxx
    rm -rf libcxx
fi

echo 'Nuttx - Apply patches'
for f in $NUTTX_PATCHES/*.patch
do 
    echo $f
    [ -f "$f" ] || break    
    ( cd $NUTTX_PATH && git apply --reject --whitespace=fix $CURRENT_DIR/$f )
done;

echo Verifying Spresense install

(source ~/spresenseenv/setup && cd spresense/sdk && make distclean && tools/config.py examples/hello && make)

if id -nG "$USER" | grep -qw "$GROUP"; then
    echo $USER belongs to $GROUP
else
    echo User needs to be added to the dialout group to access device
    echo Run the following command
    echo   sudo usermod -a -G dialout \$USER
    echo Then, logout and login, to join group
fi