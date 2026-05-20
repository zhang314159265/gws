#!/bin/bash

set -ex

cd ~/gws/infer-from-scratch/

PYTHONPATH=common:generated:. time python -m main $@
