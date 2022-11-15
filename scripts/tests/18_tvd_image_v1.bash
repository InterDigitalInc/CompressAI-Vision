#!/usr/bin/env bash
# test importing tvd_image_v1 custom dataset
echo
echo 18
echo
# TODO: for unit testing image, some dummy dataset must be included
dir="/media/sampsa/4d0dff98-8e61-4a0b-a97e-ceb6bc7ccb4b/datasets/tvd"
compressai-vision import-custom --y --dataset-type=tvd-image-v1 \
--dir=$dir
