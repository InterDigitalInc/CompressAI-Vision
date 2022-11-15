#!/usr/bin/env bash
# test importing flir_v1 custom dataset
echo
echo 19
echo
# TODO: for unit testing image, some dummy dataset must be included
dir="/media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/flir"
compressai-vision import-custom --y --dataset-type=flir-image-rgb-v1 \
--dir=$dir
