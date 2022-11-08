#!/bin/bash
## https://nbconvert.readthedocs.io/en/latest/removing_cells.html

if [ $# -lt 1 ]; then
    dirnames="fiftyone download convert detectron2 evaluate encdec cli_tutorial_1 cli_tutorial_2 cli_tutorial_3 cli_tutorial_4 cli_tutorial_5 cli_tutorial_6"
else
    dirnames=$@
fi

echo $dirnames
# exit 2

for dirname in $dirnames
do
    #cd $dirname
    # jupyter nbconvert --to rst $dirname"_nb.ipynb" \
    jupyter nbconvert --to rst --template tuto_rst $dirname"_nb.ipynb" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['remove_cell']" \
    --TagRemovePreprocessor.remove_input_tags="['remove_input']" \
    --TemplateExporter.extra_template_basedirs=./templates
    #cd ..
done
# --HighlightMagicsPreprocessor.enabled=True

# substitute ! in front of CLI commands with a space
for fname in cli_*_nb.rst
do
    echo $fname
    sed -i -r "s/ \!/ /g" $fname
done

# we'd like to have these code blocks in the converted .rst files:
# for bash cells:
# .. code:: bash
# for bash output:
# .. code-block:: text
#
# but we get ".. parsed-literal::" and "code:: ipython3" instead
#
# no idea which part of the pipeline writes that "..parsed-literal::" into the rst file
#
# docs: "Under the hood, nbconvert uses pygments to highlight code"
# nbconvert: notebook (using pygments?) -> rst
# ..nopes can't be.  It must be nbconvert without pygments that generates rst
# look here:
# https://nbconvert.readthedocs.io/en/latest/customizing.html#where-are-nbconvert-templates-installed
# -> conversion from ipynb to rst is defined in a template file
# look into: ~/.local/share/jupyter/nbconvert/templates/rst/
#
# ok, now there's
# templates/tuto_rst/
# that fixes the problem
# using tag "bash" with input cells formats their input as bash
