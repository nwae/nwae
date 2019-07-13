#!/bin/bash

#
# Command line parameters
#
NLOAD=1

for keyvalue in $@; do
    echo "Key value pair [$keyvalue]"
    IFS='=' # space is set as delimiter
    read -ra KV <<< "$keyvalue" # str is read into an array as tokens separated by IFS

    if [ "$KV" == "nload" ] ; then
        NLOAD="${KV[1]}"
        echo "Set number of loads to run to $NLOAD."
    fi
done

echo "Loadtest $NLOAD times..."
x=1

while [ $x -le "$NLOAD" ]
do
	echo $x
	x=$(( $x + 1 ))
	# stdout & stderr goes into black hole
	./loadtest.sh 1>/dev/null 2>/dev/null &
done

echo "Number of Intent Engine Load Test running ="
ps ax | grep -i loadtest | wc -l
