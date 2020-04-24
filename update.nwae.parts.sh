#!/bin/bash

#
# Param 1: Source Folder of files to be copied
# Param 2: Target Folder
# Param 3: File extension to be copied. E.g. ".py"
# Param 4: Print depth, just cosmetic purpose
#
copy_python_files() {
  local folder_from="$1"
  local folder_to="$2"
  local extension="$3"
  local depth="$4"
  local next_depth="$depth-"

  echo "$depth Copying python files from $folder_from.."
  cd "$folder_from" || exit 1

  # files_in_folder="$(ls *)"
  # echo "$depth Files in folder $folder: $files_in_folder"
  for f in ls * ; do
    file_type=$(file "$f" | sed s/".*[ ]"//g)
    if [ "$file_type" == "text" ] ; then
      if [ "$(echo $f | sed s/".*[.]"//g)" == "$extension" ] ; then
        if [ "$f" == "setup.py" ] ; then
          echo "$depth IGNORE setup.py"
          continue
        fi
        if ! [ -d "$folder_to" ] ; then
          echo "$depth Folder $folder_to do not exist, creating.."
          mkdir "$folder_to"
        else
          echo "$depth OK Folder $folder_to exists"
        fi
        echo "$depth !!Copying file $f type $file_type to $folder_to/"
        cp "$f" "$folder_to/"
      fi
    elif [ "$file_type" == "directory" ] ; then
      copy_python_files "$f" "$folder_to/$f" "$extension" "$next_depth"
    # else
    #   echo "$depth Ignoring $fld, not a folder, but $file_type."
    fi
  done

  cd .. || exit 1
}

projects="
   ../nwae.math
   ../nwae.lang
   ../nwae.ml
"

for prj in $projects ; do
   echo "Copying $prj/src/nwae/* to /usr/local/git/nwae/nwae/src"
   copy_python_files "$prj/src" "/usr/local/git/nwae/nwae/src" "py" ""
done

