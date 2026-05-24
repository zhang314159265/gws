#!/bin/bash

set -ex

cd ~/gws/infer-from-scratch/qwen3-coder-30b

PYTHONPATH=../common time python -m main $@
