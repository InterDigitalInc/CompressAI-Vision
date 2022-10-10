#!/bin/bash
# dirnames="cli_tutorial_1 cli_tutorial_2 cli_tutorial_3 cli_tutorial_4 cli_tutorial_5"
dirnames="cli_tutorial_1"
for dirname in $dirnames
do
    fname=$dirname"_nb.ipynb"
    echo 
    echo EXECUTING $fname
    echo
    jupyter nbconvert --debug --to notebook --inplace --execute $fname
done
