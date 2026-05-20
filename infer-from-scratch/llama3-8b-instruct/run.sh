#!/bin/bash

set -ex

cd ~/gws/infer-from-scratch/llama3-8b-instruct
PYTHONPATH=../common time python -m main $@
