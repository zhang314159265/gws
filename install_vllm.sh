#!/bin/bash

python use_existing_torch.py
CCACHE_NOHASHDIR="true" pip install -e . --no-build-isolation -v
