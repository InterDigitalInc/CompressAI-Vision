#!/bin/bash
# you would run this script in the directory above this one
# i.e. with
# scripts/change_fo_version.bash 0.18.0
#
if [ $# -ne 1 ]; then
  echo "give version string, i.e. 0.17.1"
  echo
  pip3 index versions fiftyone
  exit 2
fi
fnames="compressai_vision/__init__.py scripts/install.bash"
for fname in $fnames
do
    sed -i -r "s/FO_VERSION\s*=.*/FO_VERSION=\"$1\"/g" $fname
done
