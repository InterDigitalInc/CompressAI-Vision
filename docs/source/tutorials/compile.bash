#!/bin/bash
# dirnames="download convert detectron2 evaluate encdec cli_tutorial_1 cli_tutorial_2 cli_tutorial_3"
# dirnames="detectron2"
dirnames="cli_tutorial_1 cli_tutorial_2 cli_tutorial_3 cli_tutorial_4 cli_tutorial_5 cli_tutorial_6"
for dirname in $dirnames
do
    #cd $dirname
    jupyter nbconvert --to rst $dirname"_nb.ipynb" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['remove_cell']" \
    --TagRemovePreprocessor.remove_input_tags="['remove_input']"
    #cd ..
done
