 
## Tutorials

Tutorials are the notebook (``*_nb.ipynb``) files in this directory.

Two scripts are provided:

[compile.bash](compile.bash) converts ipynb files into rst that can be included into the docs.

Remember that after running this, you still need to do ``make html`` in the upper-level directory.

However, converting notebooks to rst should *not* be made part of the automatic doc build process

[run.bash](run.bash) runs all the notebooks.  Carefull with this!

Both scripts accept as a parameter a single notebook name or various names.  Names should be
without the ``_nb.ipynb`` termination, i.e. just ``cli_tutorial_1``, ``detectron2``, etc.

If no argument is provided, all notebooks are done.


