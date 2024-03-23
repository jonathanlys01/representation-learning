import torch
from tqdm import tqdm
from model import AutoEncoder, ConvAutoencoder, VariationalAutoEncoder, PCA
from dataset import get_mnist, get_cifar
import os
from utils import get_device, KL_divergence

import numpy as np
import matplotlib.pyplot as plt

import argparse


def train_ae(name, n_epochs, type, noise=False, dataset_name="mnist", save=True, batch_size=64):
    
    num_channels = 1 if dataset_name == "mnist" else 3
    img_side = 28 if dataset_name == "mnist" else 32
    
    
    if type == "ae":
        model = AutoEncoder(num_channels, img_side, 100)
    elif type == "conv":
        model = ConvAutoencoder(num_channels)
    elif type == "pca":
        model = PCA(num_channels, img_side, 100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCELoss()
    
    if type == "conv":
        temp = get_mnist(batch_size=0, loader=False) if dataset_name == "mnist" else get_cifar(batch_size=0, loader=False, type="train")
        temp = next(iter(temp))
        print(model.get_shape(temp[0].unsqueeze(0)))
    
    dataset = get_mnist(batch_size=batch_size, shuffle=True) if dataset_name == "mnist" else get_cifar(batch_size=batch_size, shuffle=True, type="test")
    
    device = get_device()
    
    model.to(device)
    
    losses = np.zeros(n_epochs)
    
    pbar = tqdm(range(n_epochs))
    
    for epoch in pbar:
        for x, _ in dataset:
            
            x = x.to(device)
            
            if noise:
                x += 0.1 * x.std() * torch.randn_like(x).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            
            losses[epoch] += loss.item()
            
        pbar.set_description(f"Loss: {losses[epoch]}")
    

    os.makedirs(f"results/{name}", exist_ok=True)
    
    if save:
        torch.save(model.state_dict(), f"results/{name}/model.pth") 
    
    print("Training complete")

    plt.plot(losses)
    plt.title("Loss")
    plt.savefig(f"results/{name}/loss.png")
    plt.close()
    
def train_vae(name, n_epochs, dataset_name, batch_size=64):
        
        num_channels = 1 if dataset_name == "mnist" else 3
        img_side = 28 if dataset_name == "mnist" else 32
        
        model = VariationalAutoEncoder(
            num_channels=num_channels,
            img_side=img_side,
            latent_dim=100
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCELoss()
        
        if dataset_name == "mnist":
            dataset = get_mnist(batch_size=batch_size, shuffle=True) 
        else:
            dataset = get_cifar(batch_size=batch_size, shuffle=True, type="train")
        
        device = get_device()
        
        model.to(device)
        
        losses = np.zeros(n_epochs)
        
        pbar = tqdm(range(n_epochs))
        
        for epoch in pbar:
            for x, _ in dataset:
                
                x = x.to(device)
                
                optimizer.zero_grad()
                output, mu, logvar = model(x)

                
                
                loss = criterion(output, x) #+ KL_divergence(mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                losses[epoch] += loss.item()
                
            pbar.set_description(f"Loss: {losses[epoch]}")
        
        pbar.set_description("Computing class prototypes")
        dataset = get_mnist(batch_size=0, loader=False) if dataset_name == "mnist" else get_cifar(batch_size=0, loader=False, type="train")
        mus = torch.zeros(10, model.latent_dim).to(device)
        logvars = torch.zeros(10, model.latent_dim).to(device)
        min_mse = torch.ones(10).to(device) * float("inf")
        min_mse = min_mse.to(device)
        
        model.eval()
        # compute class prototypes (best reconstruction for each class)
        with torch.inference_mode():
            for x, y in tqdm(dataset, total=len(dataset)):
                x = x.to(device).unsqueeze(0)
                mu, logvar = model.encode(x)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                x_hat = model.decode(z)
                mse = torch.mean((x.flatten() - x_hat.flatten()) ** 2)
                if mse < min_mse[y]:
                    min_mse[y] = mse
                    mus[y] = mu
                    logvars[y] = logvar
        
        params = {
            "mus": mus,
            "logvars": logvars
        }
    
        os.makedirs(f"results/{name}", exist_ok=True)
        
        torch.save(model.state_dict(), f"results/{name}/model.pth") 
        torch.save(params, f"results/{name}/params.pth")
        
        print("Training complete")
    
        plt.plot(losses)
        plt.title("Loss")
        plt.savefig(f"results/{name}/loss.png")
        plt.close()    

if __name__ == '__main__':
    
    # python3 train.py -m ae -n 30 -na ae1 -no -d cifar
    # trains an autoencoder for 30 epochs, with noise, and saves the model in results/ae1
    
    
    # python3 train.py --model pca --n_epochs 100 --name pca --dataset cifar --batch_size 512   
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ae", help="Model to train", 
                        choices=["ae", "vae", "conv", "pca"])
    
    parser.add_argument("--n_epochs", "-n", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--name", "-na", type=str, help="Name of the model")
    parser.add_argument("--noise", "-no", action="store_true", help="Add noise to the input") # false by default
    parser.add_argument("--dataset", "-d", type=str, default="cifar", help="Dataset to use", choices=["mnist", "cifar"])
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    if args.model in ["ae", "conv", "pca"]:
        train_ae(
            name=args.name,
            n_epochs=args.n_epochs,
            type=args.model,
            noise=args.noise,
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )

    else:
        train_vae(
            name=args.name,
            n_epochs=args.n_epochs,
            dataset_name=args.dataset,
            batch_size=args.batch_size
        )
        
        
    print("Done")
