#!/bin/bash

MODULEDIR=`pwd | sed s/.*[/]//g`

TOPDIR=`pwd | sed s/[/]$MODULEDIR//g`
echo "Using module directory $MODULEDIR and top directory $TOPDIR."

export PYTHONIOENCODING=utf-8

PYTHONPATH="$TOPDIR"/"$MODULEDIR" \
   /usr/bin/python3.6 \
   ie/api/IntentApi.py \
     topdir=$TOPDIR
