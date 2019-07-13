#!/bin/bash

COMPILE_DIR="ie"

source ../../mozg.common/src/run.common.sh

export PYTHONIOENCODING=utf-8

PYTHONPATH="$PYTHON_BIN"/"$MODULEDIR":"$COMMONSRC" \
   $PYTHON_BIN -m ie.app.chatbot.BotTest \
      topdir=$PROJECTDIR "$@"
