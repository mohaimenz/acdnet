if [ ! -d libcxx ]; then
  current_dir=`pwd`
  git clone https://bitbucket.org/acassis/libcxx --depth=1
  (cd libcxx; ./install.sh $current_dir/spresense/nuttx)
  cp patches/optional.cxx $current_dir/spresense/nuttx/libs/libxx/libcxx
  rm -rf libcxx
fi

