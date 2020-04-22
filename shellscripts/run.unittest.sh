#!/bin/bash

#
# From /server.scripts folder, run
#   ./run.unittest.sh
#

# Script name is the first parameter on command line
SCRIPT_NAME="$0"

########################################################################################
# Modify only these variables
########################################################################################
# Where this script is relative to project directory
PROGRAM_NAME="NWAE UNIT TEST"
SCRIPT_DIR="shellscripts"
PYTHON_VER="3.6"
SOURCE_DIR="../src"
COMPILE_MODULE="."
MODULE_TO_RUN="nwae.lib.NwaeUnitTest"
# Folders separated by ":"
EXTERNAL_SRC_FOLDERS="../../../nwae/nwae/src:../../../nwae/nwae.utils/src:../../../nwae/mex/src"
########################################################################################

#
# Command line parameters
#
PORT=
CONFIGFILE=

#
# For gunicorn stuff
#
for keyvalue in "$@"; do
    echo "[$SCRIPT_NAME] Key value pair [$keyvalue]"
    IFS='=' # space is set as delimiter
    read -ra KV <<< "$keyvalue" # str is read into an array as tokens separated by IFS
    if [ "$KV" == "port" ] ; then
        PORT="${KV[1]}"
        echo "[$SCRIPT_NAME] Set port to $PORT."
    elif [ "$KV" == "configfile" ] ; then
        CONFIGFILE="${KV[1]}"
        echo "[$SCRIPT_NAME] Set configfile to $CONFIGFILE."
    fi
done

#if [ "$PORT" = "" ]; then
#  echo "[$SCRIPT_NAME] ERROR Port not specified on command line. Exit 1."
#  exit 1
#fi

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
# Convert to gigabytes
RAM_MEMORY_GB="$((RAM_MEMORY / (1024*1024)))"
echo "[$SCRIPT_NAME] Total RAM $RAM_MEMORY_GB GB."

#
# This is the project directory
#
PROJECTDIR=$(pwd | sed s/[/]"$SCRIPT_DIR"//g)
echo "[$SCRIPT_NAME] Using project directory $PROJECTDIR."

for ext_src_folder in $(echo "$EXTERNAL_SRC_FOLDERS" | sed s/:/ /g) ; do
  echo "[$SCRIPT_NAME] Checking external src directory $ext_src_folder"
  if ! ls $ext_src_folder 1>/dev/null; then
    echo "[$SCRIPT_NAME] No such directory $ext_src_folder"
    exit 1
  else
    echo "[$SCRIPT_NAME] OK Directory $ext_src_folder"
  fi
done

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

#
# Get command line params
#
echo "[$SCRIPT_NAME] Command line params: [$*]"

#
# Go into source directory
#
if ! cd $SOURCE_DIR ; then
  echo "[$SCRIPT_NAME] ERROR Cannot change to source directory $SOURCE_DIR."
  exit 1
else
  echo "[$SCRIPT_NAME] OK Now in source directory '$(pwd)'."
fi

#
# Compile to byte code first
#
echo "[$SCRIPT_NAME] Compiling to byte code in module '$COMPILE_MODULE'..."
if ! $PYTHON_BIN -m compileall "$COMPILE_MODULE"; then
    echo "[$SCRIPT_NAME] ERROR Failed compilation!"
    exit 1
else
    echo "[$SCRIPT_NAME] OK Compilation to byte code successful"
fi

export PYTHONIOENCODING=utf-8

exit_status=0
echo "[$SCRIPT_NAME] Starting $PROGRAM_NAME.."
if PYTHONPATH="$PROJECTDIR"/"$SOURCE_DIR":"$EXTERNAL_SRC_FOLDERS" \
      $PYTHON_BIN -m "$MODULE_TO_RUN" \
         configfile="$CONFIGFILE" \
         port="$PORT"; then
  exit_status=0
  echo "[$SCRIPT_NAME] OK Unit Tests."
else
  echo "[$SCRIPT_NAME] FAIL Unit Tests."
  exit_status=1
fi

if ! cd - ; then
  echo "[$SCRIPT_NAME] ERROR Can't change back to original folder."
  exit 1
fi

exit "$exit_status"
