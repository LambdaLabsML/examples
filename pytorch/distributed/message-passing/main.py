import os
import argparse

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])


def run(rank, backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(rank))
        tensor = tensor.to(device)

    if rank == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('Rank {} sent data  to Rank {}'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('Rank {} has received data from rank {}'.format(rank, 0))


def init_processes(local_rank, backend):
    dist.init_process_group(backend, rank=local_rank)
    run(local_rank, backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(LOCAL_RANK, backend=args.backend)