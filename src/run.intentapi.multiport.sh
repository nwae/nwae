#!/bin/bash

COMPILE_DIR="ie"

source ../../mozg.common/src/run.common.sh

RUN_PORTS="5000 5001 5002 5003"

export PYTHONIOENCODING=utf-8

for port in $RUN_PORTS
do
    echo "Start Intent Engine on port $port.."
    PYTHONPATH="$PROJECTDIR"/"$MODULEDIR":"$COMMONSRC" \
       $PYTHON_BIN -m ie.api.IntentApi \
         topdir=$PROJECTDIR \
         port=$port \
         "$@" &
    disown
done
