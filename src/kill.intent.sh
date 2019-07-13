#!/bin/bash

echo "Processes found.."
intent_processes=`ps ax | grep -i "gunicorn" | grep "mozg.nlp" | sed s/"^ "//g`
echo "$intent_processes"

for prc in `echo "$intent_processes" | sed s/" .*"//g`
  do
    echo "Killing $prc..."
    kill $prc
  done
