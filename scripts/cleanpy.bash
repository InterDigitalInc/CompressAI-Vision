#!/bin/bash
## most of these should be done through .gitignore
## clean python bytecode
find . -name "__pycache__" -exec rm -rf {} \;
find . -name "*.pyc" -exec rm -rf {} \;
## python autodoc clean:
find . -name "*.pickle" -exec rm -rf {} \;
## clean python build
rm -rf dist
rm -rf build
# rm -rf *.egg-info
rm -f *.deb
