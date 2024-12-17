#!/bin/bash

# Example output:
# ========
# OOB
# 5.178ms    4.295GB    829.52GB/s
# MA
# 5.185ms    4.295GB    828.34GB/s
# MA+CDT
# 3.292ms    4.295GB    1304.70GB/s
# ========

DIR=$(dirname $0)

pyfile=$DIR/k.py

echo "OOB"
python $pyfile

echo "MA"
TORCHINDUCTOR_MAX_AUTOTUNE=1 python $pyfile

echo "MA+CDT"
TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python $pyfile
