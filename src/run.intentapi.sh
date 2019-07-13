#!/bin/bash

COMPILE_DIR="ie"

source ../../mozg.common/src/run.common.sh

#
# Command line parameters
#
PORT=5000
DEBUG=0
MINIMAL=0
TRAINING=0

for keyvalue in $@; do
    echo "Key value pair [$keyvalue]"
    IFS='=' # space is set as delimiter
    read -ra KV <<< "$keyvalue" # str is read into an array as tokens separated by IFS

    if [ "$KV" == "debug" ] ; then
        DEBUG=${KV[1]}
        echo "Set debug flag to $DEBUG."
    elif [ "$KV" == "port" ] ; then
        PORT=${KV[1]}
        echo "Set port to $PORT."
    elif [ "$KV" == "minimal" ] ; then
        MINIMAL=${KV[1]}
        echo "Set minimal to $MINIMAL."
    elif [ "$KV" == "training" ] ; then
        TRAINING=${KV[1]}
        echo "Set training to $TRAINING."
    fi
done

export PYTHONIOENCODING=utf-8

PYTHONPATH="$PROJECTDIR"/"$MODULEDIR":"$COMMONSRC" \
   $PYTHON_BIN -m ie.api.IntentApi \
     topdir="$PROJECTDIR" \
     port="$PORT" \
     minimal="$MINIMAL" \
     training="$TRAINING" \
     debug="$DEBUG"
