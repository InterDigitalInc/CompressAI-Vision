#!/usr/bin/env bash

# tests subcommands manual, info & list
# https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html
set -e

echo
echo 03
echo
echo
echo 03 DOWNLOAD
echo
compressai-vision download --y \
--dataset-name=quickstart \
--dir=/tmp/compressai-vision/quickstart
#
echo
echo 03 REGISTER
echo
compressai-vision register --y \
--dataset-name=quickstart-2 \
--dir=/tmp/compressai-vision/quickstart \
--type=FiftyOneDataset
#
echo
echo 03 DUMMY
echo
compressai-vision dummy --y \
--dataset-name=quickstart-2
#
echo
echo 03 DEREGISTER
echo
compressai-vision deregister --y \
--dataset-name=quickstart-2 \
#
echo
echo 03 LIST
echo
compressai-vision list
#
