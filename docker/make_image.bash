#!/bin/bash
## usage: 
# ./make_image.bash 1
## --> creates image from Dockerfile.1 & tags it with name compressai_vision:1
docker build -f Dockerfile.$1 --tag compressai_vision:$1 . $2
