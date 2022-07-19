# Examples of PyTorch Multi-Node Distributed Applications


# Table of Contents
1. [Mssage Passing](#message-passing)
    1. [torch.distributed.launch](#torch.distributed.launch)
    2. [mpirun](#mpirun)
2. [ResNet50](#resnet50)
    1. [torch.distributed.launch](#torch.distributed.launch)
    2. [mpirun](#mpirun)
3. [Tacotron2](#tacotron2)
    1. [torch.distributed.launch](#torch.distributed.launch)
    2. [mpirun](#mpirun)

## Message Passing

### torch.distributed.launch

Single-Node, Multi-GPUs
```
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=localhost --master_port=1234 \
message-passing/main.py \
--backend=nccl
```

Multi-Nodes, Multi-GPUs
```
# On the first (master) node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr="ip-of-master-node" --master_port=1234 \
message-passing/main.py \
--backend=nccl

# On the second node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr="ip-of-master-node" --master_port=1234 \
message-passing/main.py \
--backend=nccl
```

The Multi-Node, Multi-GPUs test hangs at the end, not sure why. This is how to kill it (run on all nodes)

```
kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}')
```


### mpirun

Single-Node, Multi-GPUs
```
mpirun -np 2 \
    -x MASTER_ADDR=localhost \
    -x MASTER_PORT=1234 \
    -x GPU_PER_NODE=2 \
    -x PATH \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 message-passing/main_mpirun.py --backend=nccl
```

Multi-Nodes, Multi-GPUs
```
# master node need to have password-less access to worker nodes

# On the master node
mpirun -np 4 \
    -H xxx.xxx.xxx.xxx:2,xxx.xxx.xxx.xxx:2 \
    -x MASTER_ADDR=xxx.xxx.xxx.xxx \
    -x MASTER_PORT=1234 \
    -x GPU_PER_NODE=2 \
    -x PATH \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 message-passing/main_mpirun.py --backend=nccl
```

## ResNet50

### torch.distributed.launch

The script is implmeneted by [Lei Mao](https://leimao.github.io/blog/PyTorch-Distributed-Training/)

Single-Node, Multi-GPUs
```
mkdir -p saved_models

python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=localhost --master_port=1234 \
resnet50/main.py \
--backend=nccl --use_syn
```

Multi-Nodes, Multi-GPUs
```
# On the first (master) node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr="ip-of-master-node" --master_port=1234 \
resnet50/main.py \
--backend=nccl --use_syn

# On the second node
python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr="ip-of-master-node" --master_port=1234 \
resnet50/main.py \
--backend=nccl --use_syn
```

### mpirun

Single-Node, Multi-GPUs
```
mpirun -np 2 \
    -x MASTER_ADDR=localhost \
    -x MASTER_PORT=1234 \
    -x GPU_PER_NODE=2 \
    -x PATH \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 resnet50/main_mpirun.py --backend=nccl --use_syn
```

Multi-Nodes, Multi-GPUs
```
mpirun -np 4 \
    -H xxx.xxx.xxx.xxx:2,xxx.xxx.xxx.xxx:2 \
    -x MASTER_ADDR=xxx.xxx.xxx.xxx \
    -x MASTER_PORT=1234 \
    -x GPU_PER_NODE=2 \
    -x NCCL_DEBUG=INFO -x PATH \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 resnet50/main_mpirun.py --backend=nccl --use_syn
```

## Tacotron2

### torch.distributed.launch

### mpirun


## Notes

`torch.distributed.launch` requires manually ssh into every worker node and run the launch command on each of them. 

`mpirun` can launch the job from a single node, but requires that node has password-less login to all other nodes.