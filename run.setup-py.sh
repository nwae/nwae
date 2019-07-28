#!/bin/bash

PYTHON=/usr/local/bin/python3.6
PIP=/usr/local/bin/pip3.6

# Create wheel
$PYTHON src/setup.py bdist_wheel

# Uninstall old mozg
$PIP uninstall mozg

# Install back
$PIP install dist/mozg-0.1.0-py3-none-any.whl

