#!/usr/bin/env bash
# test importing sfu-hw-objects-v1 custom dataset
echo
echo 16
echo
# TODO: test also video-convert
# TODO: for unit testing image, some dummy dataset must be included
dir="/home/sampsa/silo/interdigital/mock/SFU-HW-Objects-v1"
compressai-vision import-custom --y --dataset-type=sfu-hw-objects-v1 \
--dir=$dir
