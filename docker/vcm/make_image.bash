#!/bin/bash
## usage:
# ./make_image.bash "v1"
## --> creates image from Dockerfile and tags it compressai_vision
docker build -f Dockerfile --tag compressai_vision:$1 .
