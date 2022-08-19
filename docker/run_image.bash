#!/bin/bash
if [ $# -eq 0 ]; then
    echo "USING JUST CPU"
    nvidia_extras=""
else
    echo "USING GPU"
    nvidia_extras="--runtime=nvidia --gpus 1"
fi

## ** EDIT THIS: WHERE YOUR NOTEBOOKS ARE **
notebook_dir="../notebook" # notebooks in this dir --> visible in docker at /mnt/notebook

## ** EDIT THIS: PYTHON MODULES YOU WANT TO HOT-RELOAD **
## check that these paths exist

#python_module_dir1="../../../CompressAI" # hot-reload code from this python package --> visible in docker at /mnt/python-module1
#python_module_dir2="../../../YOLOX" # hot-reload code from this python package --> visible in docker at /mnt/python-module2
python_module_dir3="../compressai_tmp" # hot-reload code from this python package --> visible in docker at /mnt/python-module3 

## NOTE: requirement.txt files of these modules should have been placed into extra_deps/ before building the docker image
## NOTE: there is a corresponding bind mount for each one of these in the docker run command (see below)

## ** EDIT THIS: DATA ETC. DIRECTORIES **
data_dir="./dummy1" # some random data of yours --> visible in docker at /mnt/data
train_dir="./dummy2" # yet another random dir --> visible in docker at /mnt/train
vtm_dir="./VVCSoftware_VTM"

## NOTE: /root/.cache/torch is where toch caches models
## mongo saves stuff into /root/.fiftyone/var/lib/mongo
## commented out these:
#--mount type=bind,source="$(pwd)"/$python_module_dir1,target=/mnt/python-module1 \
#--mount type=bind,source="$(pwd)"/$python_module_dir2,target=/mnt/python-module2 \
## final command:
#cd /mnt/python-module1 && pip3 install --user --no-deps -e . && \
#cd /mnt/python-module2 && pip3 install --user --no-deps -e . && \
docker run --cap-add all $nvidia_extras -it -p 8889:8888 -p 8890:8890 \
--mount type=bind,source="$(pwd)"/$notebook_dir,target=/mnt/notebook \
--mount type=bind,source="$(pwd)"/$python_module_dir3,target=/mnt/python-module3 \
--mount type=bind,source="$(pwd)"/$data_dir,target=/mnt/data \
--mount type=bind,source="$(pwd)"/$train_dir,target=/mnt/train \
--mount type=bind,source="$(pwd)"/$vtm_dir,target=/mnt/VVCSoftware_VTM \
--mount source=torch_vol,target=/root/.cache/torch \
--mount source=fo_vol,target=/root/.fiftyone/var/lib/mongo \
--ipc=host \
my-interdigital-ctc \
# /bin/bash -c '\
# echo Creating links for hot-reloading python modules: this takes a few seconds && \
# cd /mnt/python-module3 && pip3 install --user --no-deps -e . && \
# tree ~/.local --filelimit=10
# rm -rf ~/.nv/ && \
# nvidia-smi & \
# # echo && \
# # echo NOTEBOOK RUNNING AT http://localhost:8889 && \
# # echo TENSORBOARD RUNNING AT http://localhost:8890 && \
# # echo && \
# # tensorboard --bind_all --port 8890 --logdir /mnt/train/runs & \
# # jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir="/mnt/notebook" \
# '
/bin/bash
# WARNING: SYMLINKS WILL NEVER SHOWN IN DOCKER CONTAINER
# https://stackoverflow.com/questions/38485607/mount-host-directory-with-a-symbolic-link-inside-in-docker-container
#
# TODO:
# that "pip3 install --user --no-deps -e ." can be skipped by first checking with
# kokkelis=$(find lib/python3.8/site-packages/ -name "*.egg-link" | grep -i "yolox" | wc -l)
# and then check value of kokkelis & doing the install only if the link has not yet been created & also caching as a volume /root/.local
#
