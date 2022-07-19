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
--master_addr="104.171.200.61" --master_port=1234 \
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


### mpirun

## ResNet50

### torch.distributed.launch

### mpirun


## Tacotron2

### torch.distributed.launch

### mpirun