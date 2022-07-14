#!/bin/sh
git clone https://github.com/EleutherAI/gpt-neox.git
cd gpt-neox
sudo apt-get install -y python3-pybind11
pip install -r requirements/requirements.txt
sudo python ./megatron/fused_kernels/setup.py install
pip install protobuf==3.20.1

HEAD_IP=$1

sudo apt install -y nfs-common
sudo mkdir ./data
sudo mount ${HEAD_IP}:${PWD}/data ./data