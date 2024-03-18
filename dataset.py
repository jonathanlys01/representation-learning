import torchvision
import torch

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
        #num_workers=2
    )
    return dataloader
    
    
if __name__ == '__main__':
    dataset = get_mnist()
    print(dataset[0][0].shape) # [1, 28, 28]