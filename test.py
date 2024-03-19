from dataset import get_mnist, get_cifar
from model import AutoEncoder, ConvAutoencoder, CondVariationalAutoEncoder, VariationalAutoEncoder
import torch
from utils import get_device
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from tqdm import tqdm
from viz import visualize

def test_auto_encoder(name, type, dataset_name="mnist", noise=False):
        
        num_channels = 1 if dataset_name == "mnist" else 3
        img_side = 28 if dataset_name == "mnist" else 32
        
        if type == "ae":
            model = AutoEncoder(num_channels, img_side, 64)
        elif type == "conv":
            model = ConvAutoencoder(num_channels)
        
        model.load_state_dict(torch.load(f"results/{name}/model.pth"))
        
        device = get_device()
        model.to(device)
        
        dataset = get_mnist(batch_size=64, loader=False) if dataset_name == "mnist" else get_cifar(batch_size=64, loader=False)
        
        for i in range(5):
            
                x, _ = dataset[i]
                
                x = x.to(device)
                
                if noise:
                    x += 0.1 * x.std() * torch.randn_like(x).to(device)
                
                output = model(x).squeeze(0)

                
                
                plt.subplot(1, 2, 1)
                plt.imshow(x.permute(1, 2, 0).cpu().numpy())
                
                plt.subplot(1, 2, 2)
                plt.imshow(output.permute(1, 2, 0).detach().cpu().numpy())
                
                plt.savefig(f"results/{name}/img_{i}.png")
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
    parser.add_argument("--dataset", "-d", type=str, default="mnist", help="Dataset to use", choices=["mnist", "cifar"])
    
    args = parser.parse_args()
    
    if args.model in ["ae", "conv"]:
        test_auto_encoder(
            name=args.name,
            type=args.model,
            dataset_name=args.dataset,
            noise="noisy" in args.name
        )
    elif args.model == "vae":
        raise NotImplementedError("VAE not implemented yet")
    