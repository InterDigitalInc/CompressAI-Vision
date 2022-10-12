#!/bin/bash

if [ $# -lt 1 ]; then
    dirnames="fiftyone download convert detectron2 evaluate encdec cli_tutorial_1 cli_tutorial_2 cli_tutorial_3 cli_tutorial_4 cli_tutorial_5 cli_tutorial_6"
else
    dirnames=$@
fi

echo $dirnames

for dirname in $dirnames
do
    fname=$dirname"_nb.ipynb"
    echo 
    echo EXECUTING $fname
    echo
    jupyter nbconvert --debug --to notebook --inplace --execute $fname
done
