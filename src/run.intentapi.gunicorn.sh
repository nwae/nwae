#!/bin/bash

COMPILE_DIR="ie"

source ../../mozg.common/src/run.common.sh

ACCOUNT=''
PORT=5000
MINIMAL=0
DEBUG=0
GUNICORN_WORKERS=2
WORKER_TYPE="sync"
WORKER_TYPE_FLAG=""

if [ $RAM_MEMORY_GB -ge 32 ] ; then
    GUNICORN_WORKERS=6
elif [ $RAM_MEMORY_GB -ge 16 ] ; then
    GUNICORN_WORKERS=4
fi

echo "Using gunicorn workers = $GUNICORN_WORKERS, for RAM $RAM_MEMORY_GB GB."

source ../../mozg.common/src/run.common.gunicorn.sh

export PYTHONIOENCODING=utf-8

PYTHONPATH="$PROJECTDIR"/"$MODULEDIR":"$COMMONSRC" \
   $GUNICORN_BIN \
      -w "$GUNICORN_WORKERS" -k "$WORKER_TYPE" $WORKER_TYPE_FLAG \
      --bind 0.0.0.0:"$PORT" \
         ie.api.Gunicorn:app \
            topdir="$PROJECTDIR" \
            account="$ACCOUNT" \
            port="$PORT" \
            minimal="$MINIMAL" \
            debug="$DEBUG"

