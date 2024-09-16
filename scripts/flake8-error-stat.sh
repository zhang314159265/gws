#!/bin/bash

if [ $# -ne 1 ]; then
  echo Need provide a directory name
  exit 1
fi

dir=$1
shift

grep -r --exclude=*.pyc --exclude=*.swp "noqa" $dir | sed 's/.*# noqa/noqa/' | grep -v "# flake8: noqa"
