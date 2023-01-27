# Set up Lambda Cloud Instances For Distributed PyTorch Training


We often receive questions about how to run multi-node distributed training on Lambda Cloud. While this [blog](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide) gives the guide for writing and running distributed PyTorch programs, we feel it is necessary to provide another tutorial that focuses on "preparing the instance" to run those programs. So let's dive into this much shorter that will hopefully help you jump-start the multi-node training on Lambda Cloud.

## Quick Start

Step one: Download the `setup_nodes.sh` and `config.sh` scripts to your __local__ machine where your cloud ssh key is stored. 

Step two: Set the variables in the `config.sh` according to your own case:

```
LAMBDA_CLOUD_KEY="path-to-your-cloud-ssh-key"
HEAD_IP="head-node-public-ip"
WORKER_IP="worker-0-public-ip worker-1-public-ip"
```

Step three: run the `setup_nodes.sh` script

```
./setup_nodes.sh
```

You will be asked to type `yes` and hit `enter` a few times. After that, you will have the minimal setup needed to run PyTorch DDP jobs on the cloud instances defined in the `config.sh` script. By "minimal" we mean each node will have its own copy of the data and code in the home directory -- so no shared storage is used (we will cover that in a separate tutorial).

You can ssh into your head node and try the resnet distributed training example we prepared for you. For example:

```
mpirun -np 3 \
-H head-node-public-ip:1,worker-0-public-ip:1,worker-1-public-ip:1 \
-x MASTER_ADDR=head-node-public-ip \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 -u examples/pytorch/distributed/resnet/main.py --backend=nccl --use_syn --batch_size=16 --arch=resnet152
```

## Explanation

The `setup_nodes.sh` script mainly does two things: give the head node passwordless access to all the nodes; disable Infiniband for NCCL (since Lambda's on-demand instance doesn't support Infiniband).

It also tests the passwordless setting by letting the head node ssh into all the nodes, including itself, and clone the example repo to the home directory of all the instances so that you can test the resnet training right away.

## How to Launch A PyTorch DDP Job

In this [tutorial](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide) we describe [three ways](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide#launch-multi-node-pytorch-distributed-applications) to launch distributed PyTorch job across multiple nodes. We briefly outline them here using the above three nodes (one gpu per node) example.

### torch.distributed.launch
PyTorch's classical (outdated?) API for launching distributed job. You need to ssh into all the nodes and launch the job from each of them:

```
# Head node
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 0
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=1 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 1
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=2 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152
```

### torchrun
PyTorch's new API for launching distributed job. Similar to `torch.distributed.launch`, you need to ssh into the nodes and launch the job from each of them (literally replacing `python3 -m torch.distributed.launch` by `torchrun`):

```
# Head node
torchrun \
--nproc_per_node=1 --nnodes=3 --node_rank=0 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 0
torchrun \
--nproc_per_node=1 --nnodes=3 --node_rank=1 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 1
torchrun \
--nproc_per_node=1 --nnodes=3 --node_rank=2 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152
```

Notice torchrun is supported by PyTorch 1.10 or newer. It also has to be searchable in the $PATH environment variable, otherwise you will see the torchrun: command not found error. We have tested torchrun on Lambda Cloud instances by creating a virtual Python environment and install the latest 1.12.1 stable PyTorch release.

```
virtualenv -p /usr/bin/python3.8 venv-torchrun
. venv-torchrun/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### mpirun
Different from the above two methods, mpirun only needs to be executed once on the head node. Notice the command has variables for all the workers which enable `mpirun` to distribute the job across multiple nodes.

```
mpirun -np 3 \
-H head-node-public-ip:1,worker-0-public-ip:1,worker-1-public-ip:1 \
-x MASTER_ADDR=head-node-public-ip \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 -u examples/pytorch/distributed/resnet/main.py --backend=nccl --use_syn --batch_size=16 --arch=resnet152
```