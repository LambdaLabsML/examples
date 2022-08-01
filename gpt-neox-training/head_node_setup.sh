#!/bin/sh
git clone https://github.com/EleutherAI/gpt-neox.git
cd gpt-neox
sudo apt-get install -y python3-pybind11
pip install -r requirements/requirements.txt
sudo python ./megatron/fused_kernels/setup.py install
pip install protobuf==3.20.1

WORKER_IP=$1
N_GPUS=$2

sudo apt install -y nfs-kernel-server
sudo mkdir ./data
sudo chmod 777 ./data
printf "${PWD}/data ${WORKER_IP}(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo systemctl restart nfs-kernel-server
python prepare_data.py -d ./data

sudo apt-get install -y pdsh
export DSHPATH=$PATH
export PDSH_RCMD_TYPE=ssh

ssh-keygen -t rsa -N ''
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

sudo mkdir /job
printf "localhost slots=$N_GPUS\n$WORKER_IP slots=$N_GPUS" | sudo tee /job/hostfile
