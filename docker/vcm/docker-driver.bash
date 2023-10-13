#!/bin/bash 
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# distribution="ubuntu18.04"
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime
sudo cp daemon.json /etc/docker/
sudo systemctl restart docker
