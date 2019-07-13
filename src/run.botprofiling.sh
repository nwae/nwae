#!/bin/bash

COMPILE_DIR="ie"

source ../../mozg.common/src/run.common.sh

export PYTHONIOENCODING=utf-8

PYTHONPATH="$PROJECTDIR"/"$MODULEDIR":"$COMMONSRC" \
   $PYTHON_BIN -m ie.app.chatbot.BotProfiling \
      topdir="$PROJECTDIR" "$0"
