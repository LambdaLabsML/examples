
import sys
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import typer

from loggers import get_logger, SUPPORTED_LOGGERS


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(data_path, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,testloader,classes

def train(num_epochs, trainloader, net, criterion, optimizer, device, logger, log_its=100):
    step = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            if i % log_its == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
                logger.log({"train/loss": loss.item()}, step=step)
            step += 1

        logger.log({"train/epoch": epoch}, step=step)

    print('Finished Training')


@torch.no_grad()
def eval(testloader, classes, net, logger, device):
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    grid = torchvision.utils.make_grid(images)
    outputs = net(images.to(device))
    _, predictions = torch.max(outputs, 1)

    logger.log_image({"test/inputs": grid})

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predictions = torch.max(outputs.data, 1)

        all_preds.append(predictions.cpu())
        all_labels.append(labels)

        total += labels.size(0)
        correct += (predictions == labels.to(device)).sum().item()

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)


    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    logger.log_metric("val/accuracy", correct/total)

    # Use an advanced logging feature e.g. plotting confusion matrix
    print(all_labels.shape, all_preds.shape, classes)
    logger.log_confusion_matrix("val/confusion", all_labels, all_preds, classes)

def do_run(
    logger,
    batch_size=128,
    lr=0.01,
    num_epochs=5,
    ):
    cfg = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
    }

    # Just save to the same place as we don't really care about them
    weights_path = './cifar_net.pth'
    data_path = "./cifar10_data"
    logger.log_cfg(cfg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader, classes = load_data(data_path, batch_size)

    logger.log_dataset_ref("cifar10", data_path)

    net = Net()
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train(num_epochs, trainloader, net, criterion, optimizer, device, logger)

    torch.save(net.state_dict(), weights_path)
    logger.log_model_weights("cifar10-model", weights_path)

    eval(testloader, classes, net, logger, device)

def main(
    log_type:str=typer.Argument("wandb", help=f"Choose from {SUPPORTED_LOGGERS}"),
    batch_size:int=128,
    lr:float=0.01,
    epochs:int=5,
    project:str="cifar",
):
    logger = get_logger(log_type)
    logger.start(project)
    do_run(logger, batch_size, lr, epochs)

if __name__ == "__main__":
    typer.run(main)

