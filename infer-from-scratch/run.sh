#!/bin/bash

set -ex

cd ~/gws/infer-from-scratch/

PYTHONPATH=generated:. time python -m main $@
