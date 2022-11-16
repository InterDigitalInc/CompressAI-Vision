 
## Tutorials

Tutorials are the notebook (``*_nb.ipynb``) files in this directory.

Please note that you can see the tags that control cell visibility, etc. in python notebook web interface with:
```
view -> cell toolbar -> tags
```
Please use these two tags:

``remove_cell`` and ``bash``.  Read also [compile.bash](compile.bash) for more documentation/observations.

Two scripts are provided:

[compile.bash](compile.bash) converts ipynb files into rst that can be included into the docs.

Remember that after running this, you still need to do ``make html`` in the upper-level directory.

However, converting notebooks to rst should *not* be made part of the automatic doc build process

[run.bash](run.bash) runs all the notebooks.  Carefull with this!

Both scripts accept as a parameter a single notebook name or various names.  Names should be
without the ``_nb.ipynb`` termination, i.e. just ``cli_tutorial_1``, ``detectron2``, etc.

If no argument is provided, all notebooks are done.

### Files

```
cli_tutorial_1_nb.ipynb
cli_tutorial_2_nb.ipynb
cli_tutorial_3_nb.ipynb
cli_tutorial_4_nb.ipynb
cli_tutorial_5_nb.ipynb
cli_tutorial_6_nb.ipynb
cli_tutorial_7_nb.ipynb

fiftyone_nb.ipynb

1: download_nb.ipynb
# 2: convert_nb.ipynb # DEPRECATED
3: detectron2_nb.ipynb
4: evaluate_nb.ipynb
5: encdec_nb.ipynb
```
