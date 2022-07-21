# Examples of PyTorch Multi-Node Distributed Applications


# Table of Contents
1. [Setup the nodes](#setup-the-nodes)
2. [Launch the application](#launch-the-application)
    1. [Message Passing](#launch-the-application)
        1. [torch.distributed.launch](#torchdistributedlaunch)
        2. [mpirun](#mpirun)


# Setup The Nodes

(Skip this section if you just want to try the [demos](#message-passing))

By definition one needs at least two computers for “multi-node” applications. Let’s just use two for this tutorial but the practice should generalize to more than two nodes. 

Also we will focus on “data parallelism”, which split the data across different worker node but runs the same operation on these splits. For other types of parallelism, e.g. model parallelism or “hybrid” (data + model) parallelism, please see the example in this [documentation](https://lambdalabs.atlassian.net/wiki/spaces/MA/pages/56786973/Multi-node+Language+Model+Training+Guide). 

For both nodes to run the same operation, **they should have the same “environment”** that is needed by the operation. To be more precise, the environment include software like drivers, libraries, and code, and to a certain degree, hardware too. You may want to avoid using different types of GPUs on the nodes otherwise the less performing one will bottleneck the process by either the memory/computation, or both.

So the goal is to set up the same environment on the nodes. This can be achieved in multiple ways:

1. Provision the nodes with the environment already inside it. This is the most efficient and the least error-prone way. 
2. If something is missing from the initial provisioning, you need to step in and finish the set up. You can do this by either directly installing the missing software/libraries on the nodes, or pull a docker image that has everything you need inside it. The later method tend to be more efficient and less error-prone. 

For example, all Lambda cloud instances are provisioned with Ubuntu OS and Lambda stack, which include CUDA, Python 3, Pytorch and more. This is a fairly sophisticated environment that runs many deep learning workload out of the box.

However, if you have a specific project that requires extra code or python libraries, then you still need to set these up by yourself on Lambda cloud instances. You can either download the code and pip install the missing packages, or create a docker image that have the code & packages pre-installed and pull it onto the nodes.

Last but not the least, if your environment is inside of a docker image, you will to first create running containers from the image on each node, and then run the application inside them. Otherwise, you can just run your application from the host OS.


# Launch The Application

After you set up the environment on all the nodes, the next step is to actually run the application. There are different ways to “launch” applications in a distributed fashion across multiple nodes, designed different vendors such as HPC veterans (Open MPI), Deep Learning framework leaders (PyTorch), or opensource A.I. communities (Horovod). Your ML code needs to be customized slightly for each method, but they more or less follows the same idea: create multiple processes with efficient message passing between the processes. 

The rest of the tutorial will cover how to use PyTorch’s `distributed.launch` method and Open MPI’s `mpirun` method to run PyTorch distributed applications across multiple nodes. We will demonstrate how these two methods work with

- A “hello world” example that does basic message passing between the nodes.
- A standard example for resnet training.

As far as setting up the machine goes, we will use two 2xA6000 Lambda Cloud instance (`104.171.200.`62 and `104.171.200.182` ). This will allow us to have 4 workers in total (two on each node). We will show how the examples work from both the host OS, and inside of docker containers (`pytorch:22.06-py3`).

## Message Passing

We will start from a very simple example that passes tensor between the workers. Let’s first see how to do this with `torch.distributed.launch`. 

### torch.distributed.launch

```jsx
import os
import argparse

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def run(backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('Rank {} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('Rank {} has received data from rank {}\n'.format(WORLD_RANK, 0))

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)
```

**Launch**

```jsx
# The application has to be launched from both nodes

# Node 104.171.200.62
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=104.171.200.62 --master_port=1234 \
message-passing/main.py \
--backend=nccl

# Node 104.171.200.182
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=104.171.200.62 --master_port=1234 \
message-passing/main.py \
--backend=nccl

# output from node 104.171.200.62
Rank 1 has received data from rank 0

Rank 0 sent data to Rank 1

Rank 0 sent data to Rank 2

Rank 0 sent data to Rank 3

# output from node 104.171.200.182
Rank 2 has received data from rank 0

Rank 3 has received data from rank 0
```

**Notes**

This application simply spins up four processes, and let process `0` send a tensor to the other three processes.

- `nproc_per_nod` defines the number of processes can be created on each node. It should equal to the number of GPUs for each node.
- `nnodes` defines the number of nodes.
- `node_rank` defines the rank of a *node.* This has to be set differently in the two commands — use `0` on the master node command, and `1` on the worker node command.
- `master_addr` and `master_port` are the IP address and port for the master node. They have to be the same for both commands because there is only one master node.
- `torch.distributed.launch` creates the processes and some important environment variables for each of them.
    - `LOCAL_RANK`: it defines the rank of a *process* on a *node*. Since each node has only two GPUs, `LOCAL_RANK` can only be `0` or `1`. It also defines which CUDA devices the process should use, via the `device = torch.device("cuda:{}".format(LOCAL_RANK))` line.
    - `WORLD_SIZE`: it defines the total number of processes. We have `2` nodes x `2` processes/node, so `WORLD_SIZE=4`. In this example, we use it to create the for loop that allows process `0` to send the data to the rest of the processes.
    - `RANK`: it define the rank of a *process* in the world (all nodes combined). Since the `WORLD_SIZE` is 4, the `RANK` can be `0`, `1`, `2`, `3`.
- `torch.distributed.launch` also passes a `--local_rank` argument to the `[main.py](http://main.py)` script. It is not used in anyway in this example, but not having it in the parser will cause a error.
- `backend` is the message passing library. PyTorch support three different backends: `nccl`, `gloo`, and `mpi` (need build from source).
- `dist.init_process_group` creates the process with the correct *global rank.*

VERY IMPORTANT: `RANK` is used by `dist.init_process_group`, and `LOCAL_RANK` is used by `to(device)`.

### mpirun

Only small changes are needed to make the `[main.py](http://main.py)` work for `mpirun`.

The first change addresses the difference between how `torch.distributed.launch` and `mpirun` set the environment variables 

```jsx
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
```

Another change is you can remove `--local_rank` since `mpirun` does not use that.

That is it! You can launch the same application via the following `mpirun` command from the master node:

```jsx
# From master node
mpirun -np 4 \
-H 104.171.200.62:2,104.171.200.182:2 \
-x MASTER_ADDR=104.171.200.62 \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 message-passing/main_mpirun.py --backend=nccl

# Output on master node
Rank 1 has received data from rank 0

Rank 0 sent data to Rank 1

Rank 0 sent data to Rank 2

Rank 2 has received data from rank 0

Rank 0 sent data to Rank 3

Rank 3 has received data from rank 0
```

**Notes**

VERY IMPORTANT: The master node needs to have password-less access to all the worker node. The way to make this happen is to run `ssh-keygen` on the master node, and copy the public key to the `~/.ssh/authorized_keys` on all the worker nodes. You only need to run this command on the master node, no matter how many nodes are used by the application. 

Some explanation for the `mpirun` options:

- `np` : defines the total number of workers (processes, or the world size)
- `H`: defines the IP address and number of processes for each node. It can be too much typing when there are too many nodes, in which case one can use a [hostfile](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile) instead.
- `-bind-to none`: specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance).
- `-map-by slot`: allows you to have a mixture of different NUMA configurations because the default behavior is to bind to the socket.
- `mca pml ob1 -mca btl ^openib`: force the use of TCP for MPI communication. This avoids many multiprocessing issues that Open MPI has with RDMA which typically results in segmentation faults.