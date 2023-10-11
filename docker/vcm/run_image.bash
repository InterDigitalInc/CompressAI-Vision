#!/bin/bash
## ./run_image 1 gpu
## 
if [ $# -lt 1 ]; then
  echo "usage: ./run_image number [gpu]"
  exit
fi
if [ $# -lt 2 ]; then
    echo "USING JUST CPU"
    nvidia_extras=""
else
    echo "USING A GPU"
    nvidia_extras="--runtime=nvidia --gpus 1"
fi

## directories from your local filesystem:

## ** EDIT THIS: WHERE YOUR NOTEBOOKS ARE **
notebook_dir="/path/to/your/notebook/directory" # notebooks in this dir --> visible in docker at /mnt/notebook
data_dir="/path/to/your/data/directory" # some random data of yours --> visible in docker at /mnt/data

## If you'd like to hot-reload a python module from your local filesystem:
# python_module_dir1="../" # hot-reload code from this python package --> visible in docker at /mnt/python-module1 
## remember to mod your Dockerfile so that requirements.txt of that python module are installed into the image
## also add this line to the docker run command:
#--mount type=bind,source="$(pwd)"/$python_module_dir1,target=/mnt/python-module1 \
## and this this the bash command:
# cd /mnt/python-module1 && pip3 install --user --no-deps -e . && \

## NOTE: /root/.cache/torch is where toch caches models, mongo saves stuff into /root/.fiftyone/var/lib/mongo
## final command:
docker run --cap-add all $nvidia_extras -it -p 8889:8888 -p 8890:8890 \
--mount type=bind,source="$(pwd)"/$notebook_dir,target=/mnt/notebook \
--mount type=bind,source="$(pwd)"/$data_dir,target=/mnt/data \
--mount source=torch_vol,target=/root/.cache/torch \
--mount source=fo_vol,target=/root/.fiftyone/var/lib/mongo \
--ipc=host \
compressai_vision:$1 \
/bin/bash -c '\
rm -rf ~/.nv/ && \
nvidia-smi & \
echo && \
echo NOTEBOOK RUNNING AT http://localhost:8889 && \
echo TENSORBOARD RUNNING AT http://localhost:8890 && \
echo && \
tensorboard --bind_all --port 8890 --logdir /mnt/train/runs & \
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir="/mnt/notebook" \
'
