#!/bin/bash
## https://nbconvert.readthedocs.io/en/latest/removing_cells.html
# dirnames="download convert detectron2 evaluate encdec cli_tutorial_1 cli_tutorial_2 cli_tutorial_3"
# dirnames="detectron2"
# dirnames="cli_tutorial_1 cli_tutorial_2 cli_tutorial_3 cli_tutorial_4 cli_tutorial_5 cli_tutorial_6"
# dirnames="fiftyone"
dirnames="detectron2"
for dirname in $dirnames
do
    #cd $dirname
    jupyter nbconvert --to rst $dirname"_nb.ipynb" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['remove_cell']" \
    --TagRemovePreprocessor.remove_input_tags="['remove_input']"
    #cd ..
done

# substitute ! in front of CLI commands with a space
for fname in cli_*_nb.rst
do
    echo $fname
    sed -i -r "s/ \!/ /g" $fname
done
