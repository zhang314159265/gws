#!/bin/bash

if [ $# -ne 1 ]; then
  echo Need provide a directory name
  exit 1
fi

dir=$1
shift

grep -r 'type: ignore' $dir | sed 's/.*\(\[.*\]\).*/\1/' | tr '[], ' '\n\n\n\n' | sort | uniq -c | sort -nr
