# Set up Lambda Cloud Instances For Distributed PyTorch Training


We received many questions regarding how to do multi-node distributed training on Lambda Cloud. While this [blog](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide) gives the guide for writing and running distributed PyTorch programs, we feel it is necessary to give another tutorial that focus on "preparing the instance" to run those programs. So let's dive in this much shorter that hopefully help you jump start the multi-node training on Lambda Cloud.

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

You will be asked to type `yes` and hit `enter` for a few times. After that you will have the minimal setting that is needed to run PyTorch DDP jobs on the cloud instances defined in the `config.sh` script. By minimal we mean each node will have its own copy of the data and code in the home directory -- so no shared storage is used (we will cover that in a seperate tutorial).

You can ssh into your head node and try the resnet distributed training example we prepared for you. For example:

```
mpirun -np 3 \
-H head-node-public-ip:1,worker-0-public-ip:1,worker-1-public-ip:1 \
-x MASTER_ADDR=head-node-public-ip \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 -u examples/pytorch/distributed/resnet/main.py --backend=nccl --use_syn --batch_size=16 --arch=resnet152"
```

## Explanation

The `setup_nodes.sh` script mainly does two things: give the headnode passwordless access to all the nodes; disable Infiniband for NCCL (since Lambda's on-demand instance doesn't support Infiniband).

It also test the passwordless setting by letting the headnode ssh into all the nodes including itself, as well as clone the example repo to the home directory of all the instances, so you can test the resnet training right away.