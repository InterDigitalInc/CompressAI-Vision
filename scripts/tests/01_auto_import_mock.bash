#!/usr/bin/env bash
# mocks import and registration of supported mpeg vcm datasets
set -e

echo
echo 01
echo

# compressai-vision mpeg-vcm-auto-import --mock --y # old version
# compressai-vision import-custom --dataset-type=oiv6-mpeg-v1 --mock
compressai-vision import-custom --y --dataset-type=oiv6-mpeg-v1
