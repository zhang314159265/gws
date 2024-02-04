#!/bin/bash

# set -ex
set -e

if [ $# -ne 2 ]; then
  echo "Usage: findsym.sh [dir] [sym]"
  exit 1
fi

dir=$1
shift

sym=$1
shift

# this is not recursive
# for lib in `ls $dir/*.so $dir/*.a`; do
# this is recursive
for lib in `find $dir -name \*.a -or -name \*.so`; do
  if nm -C $dir/$lib | grep "$sym"; then
    echo "=> Found in lib $lib"
  fi
done

echo "bye"
