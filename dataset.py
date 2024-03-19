import torchvision
import torch

import matplotlib.pyplot as plt

import os 

# MNIST dataset

def get_mnist(batch_size=32, shuffle=True, loader=True):
    dataset =  torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    if not loader:
        return dataset
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= 2 if os.cpu_count() > 2 else 1
    )
    return dataloader

def get_cifar(batch_size=32, shuffle=True, loader=True):
    dataset =  torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    if not loader:
        return dataset
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers= 2 if os.cpu_count() > 2 else 1
    )
    return dataloader
    
    
if __name__ == '__main__':
    dataset = get_cifar(loader=False)
    img, label = dataset[1]
    img = img.permute(1, 2, 0)
    plt.imshow(img.numpy())
    plt.show()
