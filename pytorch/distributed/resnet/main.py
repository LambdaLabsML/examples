import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
import time
import importlib

if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    # Environment variables set by mpirun
    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
else:
    import sys
    sys.exit("Can't find the evironment variables for local rank")

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():

    num_epochs_default = 10000
    batch_size_default = 256
    image_size_default = 224
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"
    steps_syn_default = 20

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--image_size", type=int, help="Size of input image.", default=image_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--backend", type=str, help="Backend for distribted training.", default='nccl', choices=['nccl', 'gloo', 'mpi'])
    parser.add_argument("--arch", type=str, help="Model architecture.", default='resnet50', choices=['resnet50', 'resnet18', 'resnet101', 'resnet152'])
    parser.add_argument("--use_syn", action="store_true", help="Use synthetic data")
    parser.add_argument("--steps_syn", type=int, help="Step per epoch for training with synthetic data", default=steps_syn_default)
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume
    backend = argv.backend
    use_syn = argv.use_syn
    w = argv.image_size
    h = argv.image_size
    c = 3
    steps_syn = argv.steps_syn

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend=backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # Encapsulate the model on the GPU assigned to the current process
    model = getattr(torchvision.models, argv.arch)(pretrained=False)

    device = torch.device("cuda:{}".format(LOCAL_RANK))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(LOCAL_RANK)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    if use_syn:
        # Synthetic data
        inputs_syn = torch.rand((batch_size, c, w, h)).to(device)
        labels_syn = torch.zeros(batch_size, dtype=torch.int64).to(device)
    else:
        # Prepare dataset and dataloader
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Data should be prefetched
        # Download should be set to be False, because it is not multiprocess safe
        train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform) 
        test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

        # Restricts data loading to a subset of the dataset exclusive to the current process
        train_sampler = DistributedSampler(dataset=train_set)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
        # Test loader does not have to follow distributed sampling strategy
        test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    times = []
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(LOCAL_RANK, epoch))
        
        # Save and evaluate model routinely
        if not use_syn:
            if epoch % 10 == 0:
                if LOCAL_RANK == 0:
                    accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                    torch.save(ddp_model.state_dict(), model_filepath)
                    print("-" * 75)
                    print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                    print("-" * 75)

        ddp_model.train()

        if use_syn:
            start_epoch = time.time()
            for count in range(steps_syn):
                optimizer.zero_grad()
                outputs = ddp_model(inputs_syn)
                loss = criterion(outputs, labels_syn)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch

            if epoch > 0:
                times.append(elapsed)
                print('num_steps_per_gpu: {}, avg_step_time: {:.4f}'.format(count, elapsed / count))              
        else:
            train_loader.sampler.set_epoch(epoch)
            start_epoch = time.time()
            count = 0
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                count += 1
            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch

            if epoch > 0:
                times.append(elapsed)
                print('num_steps_per_gpu: {}, avg_step_time: {:.4f}'.format(count, elapsed / count))

    avg_time = sum(times) / (num_epochs - 1)

    print("Average epoch time: {}".format(avg_time))

if __name__ == "__main__":
    
    main()
