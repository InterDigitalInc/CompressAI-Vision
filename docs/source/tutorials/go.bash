#!/bin/bash
./compile.bash $@
save=$PWD
cd ../..
make html
cd $save
