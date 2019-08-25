#!/bin/bash

PYTHON=/usr/local/bin/python3.6
PIP=/usr/local/bin/pip3.6

# Clear build folder
rm -rf ./build/*

# Create wheel
$PYTHON src/setup.py bdist_wheel

# Uninstall old nwae
$PIP uninstall nwae

# Install back
$PIP install dist/nwae-0.1.0-py3-none-any.whl
