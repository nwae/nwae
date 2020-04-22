#!/bin/bash

projects="
   ../nwae.math
   ../nwae.lang
   ../nwae.ml
"

for prj in $projects ; do
   echo "Copying $prj/src/nwae/* to src/nwae/"
   cp -R $prj/src/nwae/* src/nwae/
done

