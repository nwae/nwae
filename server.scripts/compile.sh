#!/usr/bin/env bash

# Script name is the first parameter on command line
SCRIPT_NAME="$0"

#
# To be defined by the calling script
#
COMPILE_DIR=""

#
# Get RAM memory
#
# Default 1GB RAM
RAM_MEMORY=1048576
# Check if command "free" is available on system
free 1>/dev/null 2>/dev/null
if [ $? -ne 0 ]; then
  echo "[$SCRIPT_NAME] WARNING. Could not determine RAM size.. defaulting to 1GB RAM."
else
  RAM_MEMORY=$(free| grep "^Mem" | sed s/"^Mem:[ \t]*"//g | sed s/"[ \t].*"//g)
fi
RAM_MEMORY_GB="$((RAM_MEMORY / (1024*1024)))"

#
# This is where the Python module directory is
#
MODULEDIR=$(pwd | sed s/.*[/]//g)

#
# This is the project directory
#
PROJECTDIR=$(pwd | sed s/[/]"$MODULEDIR"//g)
echo "[$SCRIPT_NAME] OK Using module directory \"$MODULEDIR\" and project directory \"$PROJECTDIR\"."

SRC_NWAE_UTILS="$PROJECTDIR"/../nwae.utils/src
echo "[$SCRIPT_NAME] OK Using nwae.utils src directory \"$SRC_NWAE_UTILS\"."

PYTHON_BIN=""
FOUND=0
#
# Look for possible python paths
#
for path in "/usr/bin/python3.6" "/usr/local/bin/python3.6" "/usr/bin/python3"; do
    echo "[$SCRIPT_NAME] Checking python path $path.."

    if ls $path 2>/dev/null 1>/dev/null; then
        echo "[$SCRIPT_NAME]   OK Found python path in $path"
        PYTHON_BIN=$path
        FOUND=1
        break
    else
        echo "[$SCRIPT_NAME]   No python in path $path"
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "[$SCRIPT_NAME]   No python binary found!!"
    exit 1
fi

#
# Get command line params
#
echo "[$SCRIPT_NAME] OK Command line params: [$*]"

#
# Compile to byte code first
#
echo "[$SCRIPT_NAME] Compiling to byte code..."
if ! $PYTHON_BIN -m compileall $COMPILE_DIR; then
    echo "[$SCRIPT_NAME] ERROR Failed compilation!"
    exit 1
fi

echo "[$SCRIPT_NAME] OK Compilation to byte code successful"
