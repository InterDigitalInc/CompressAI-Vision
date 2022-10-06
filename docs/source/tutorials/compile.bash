#!/bin/bash
dirnames="download convert detectron2 evaluate encdec cli_tutorial_1 cli_tutorial_2 cli_tutorial_3"
# dirnames="detectron2"
for dirname in $dirnames
do
    #cd $dirname
    jupyter nbconvert --to rst $dirname"_nb.ipynb" --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="['remove_cell']"
    #cd ..
done
