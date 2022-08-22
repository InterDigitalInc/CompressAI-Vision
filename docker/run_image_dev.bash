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

notebook_dir="../../siloai-playground/sampsa/notebook"
data_dir="../../siloai-playground/sampsa/nokia"

## If you'd like to hot-reload a python module from your local filesystem:
python_module_dir1="../" # hot-reload code from this python package --> visible in docker at /mnt/python-module1 

## NOTE: /root/.cache/torch is where toch caches models, mongo saves stuff into /root/.fiftyone/var/lib/mongo
## final command:
docker run --cap-add all $nvidia_extras -it -p 8889:8888 -p 8890:8890 \
--mount type=bind,source="$(pwd)"/$notebook_dir,target=/mnt/notebook \
--mount type=bind,source="$(pwd)"/$data_dir,target=/mnt/data \
--mount type=bind,source="$(pwd)"/$python_module_dir1,target=/mnt/python-module1 \
--mount source=torch_vol,target=/root/.cache/torch \
--mount source=fo_vol,target=/root/.fiftyone/var/lib/mongo \
--ipc=host \
compressai_vision:$1 \
/bin/bash -c '\
cd /mnt/python-module1 && pip3 install --user --no-deps -e . && \
rm -rf ~/.nv/ && \
nvidia-smi & \
echo && \
echo NOTEBOOK RUNNING AT http://localhost:8889 && \
echo TENSORBOARD RUNNING AT http://localhost:8890 && \
echo && \
tensorboard --bind_all --port 8890 --logdir /mnt/train/runs & \
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir="/mnt/notebook" \
'
