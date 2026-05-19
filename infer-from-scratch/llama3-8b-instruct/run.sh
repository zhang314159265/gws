#!/bin/bash

set -ex

cd ~/gws/infer-from-scratch/llama3-8b-instruct
time python -m main $@
