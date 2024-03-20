from dataset import get_mnist, get_cifar, classes_cifar, classes_mnist
from model import AutoEncoder, ConvAutoencoder, VariationalAutoEncoder
import torch
from utils import get_device
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from tqdm import tqdm
from viz import visualize


def test_auto_encoder(name, type, dataset_name="mnist", noise=False, num_samples=5):
        
        num_channels = 1 if dataset_name == "mnist" else 3
        img_side = 28 if dataset_name == "mnist" else 32
        
        if type == "ae":
            model = AutoEncoder(num_channels, img_side, 64)
        elif type == "conv":
            model = ConvAutoencoder(num_channels)
        
        model.load_state_dict(torch.load(f"results/{name}/model.pth", map_location=torch.device('cpu')))
        
        device = get_device()
        model.to(device)
        
        dataset = get_mnist(batch_size=64, loader=False) if dataset_name == "mnist" else get_cifar(batch_size=64, loader=False)
        
        for i in range(num_samples):
            
                idx = random.randint(0, len(dataset))
            
                x, _ = dataset[idx]
                
                x = x.unsqueeze(0).to(device)
                
                if noise:
                    x += 0.1 * x.std() * torch.randn_like(x).to(device)
                
                output = model(x).squeeze(0)
                
                plt.subplot(1, 2, 1)
                plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
                
                plt.subplot(1, 2, 2)
                plt.imshow(output.permute(1, 2, 0).detach().cpu().numpy())
                
                plt.savefig(f"results/{name}/img_{i}.png")
                plt.close()

def test_vae(name, dataset_name):
    num_channels = 1 if dataset_name == "mnist" else 3
    img_side = 28 if dataset_name == "mnist" else 32
    
    classes = classes_mnist if dataset_name == "mnist" else classes_cifar
    
    model = VariationalAutoEncoder(num_channels, img_side)
    
    model.load_state_dict(torch.load(f"results/{name}/model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    params = torch.load(f"results/{name}/params.pth", map_location=torch.device("cpu"))
    
    device = get_device()
    
    mus = params["mus"].to(device)
    logvars = params["logvars"].to(device)
    
    
    
    model.to(device)
    
    with torch.inference_mode():
        for i in range(10): # num classes
            mu = mus[i].unsqueeze(0)
            logvar = logvars[i].unsqueeze(0)
            std = torch.exp(0.5 * logvar)
            
            z = mu + std * torch.randn_like(mu)
        
            output = model.decode(z).squeeze(0)
            
            plt.imshow(output.permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f"Perturbed prototype for class: {classes[i]}")
            plt.savefig(f"results/{name}/prototype_{i}.png")
            plt.close()
        
    
if __name__ == '__main__':
    
    # python3 test.py -m ae -n ae_noisy
    # python3 test.py -m ae -n ae
    
    # python3 test.py -m conv -n conv_noisy
    # python3 test.py -m conv -n conv
    
    # python3 test.py -m vae -n vae
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", "-n", type=str, default="ae", help="Name of the model to test")
    parser.add_argument("--model", "-m", type=str, default="ae", help="Type of model to test", choices=["ae", "conv", "vae"])
    parser.add_argument("--dataset", "-d", type=str, default="cifar", help="Dataset to use", choices=["mnist", "cifar"])
    parser.add_argument("--num_samples", "-ns", type=int, default=5, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    print("DS:", args.dataset)
    
    if args.model in ["ae", "conv"]:
        test_auto_encoder(
            name=args.name,
            type=args.model,
            dataset_name=args.dataset,
            noise="noisy" in args.name,
            num_samples=args.num_samples    
        )
    elif args.model == "vae":
        test_vae(
            name=args.name,
            dataset_name=args.dataset
        )
    print("Testing complete")
    