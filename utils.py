import torch
import torch.functional as F

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
    
def KL_divergence(mu, logvar):
    # expression for KL divergence in the case of a VAE with 
    # - Gaussian target distribution
    # - Normal distribution as the approximate posterior    
    
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())