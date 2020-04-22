#!/bin/bash

projects="
   ../nwae.math
   ../nwae.lang
   ../nwae.ml
"

for prj in $projects ; do
   echo "Copying $prj/src to src/.."
done

