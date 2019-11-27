#!/bin/bash

SCRIPT_NAME="$0"
ALL_PATH="../../nwae.utils/src:."
PYTHON_VER="3.6"

PYTHON_BIN=""
FOUND=0
#
# Look for possible python paths
#
for path in "/usr/bin/python$PYTHON_VER" "/usr/local/bin/python$PYTHON_VER"; do
    echo "[$SCRIPT_NAME] Checking python path $path.."

    if ls $path 2>/dev/null 1>/dev/null; then
        echo "[$SCRIPT_NAME]   OK Found python path in $path"
        PYTHON_BIN=$path
        FOUND=1
        break
    else
        echo "[$SCRIPT_NAME]   ERROR No python in path $path"
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "[$SCRIPT_NAME]   ERROR No python binary found!!"
    exit 1
fi

if [ "$1" = "" ]; then
  echo "No module specified!"
  exit 1
fi

PYTHONPATH="$ALL_PATH" \
  "$PYTHON_BIN" \
  $1
