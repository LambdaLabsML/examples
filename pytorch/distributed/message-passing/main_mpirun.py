import os
import argparse
import torch
import torch.distributed as dist


def run(world_rank, world_size, backend):
    tensor = torch.zeros(1)

    gpu_per_node = int(os.environ['GPU_PER_NODE'])
    local_rank = world_rank % gpu_per_node

    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(local_rank))
        tensor = tensor.to(device)

    if world_rank == 0:
        for rank_recv in range(1, world_size):
            dist.send(tensor=tensor, dst=rank_recv)
            print('Rank {} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('Rank {} has received data from rank {}\n'.format(world_rank, 0))


def init_processes(world_rank, world_size, backend):
    dist.init_process_group(backend, rank=world_rank, world_size=world_size)
    run(world_rank, world_size, backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_processes(world_rank, world_size, backend=args.backend)
