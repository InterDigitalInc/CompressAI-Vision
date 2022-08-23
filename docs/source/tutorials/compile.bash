#!/bin/bash
dirnames="download convert detectron2 evaluate encdec"
# dirnames="detectron2"
for dirname in $dirnames
do
    #cd $dirname
    jupyter nbconvert --to rst $dirname"_nb.ipynb" --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="['remove_cell']"
    #cd ..
done
