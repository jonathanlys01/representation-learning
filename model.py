import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    
    def __init__(
        self,
        data_size: int,
        hidden_size: int,
    ):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Linear(data_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, data_size)
        
    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self, num_channels=1):
        super(ConvAutoencoder, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2) 
        self.t_conv2 = nn.ConvTranspose2d(16, num_channels, 2, stride=2)
        
        
        

    def forward(self, x, verbose=False):    

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if verbose: print(x.shape)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        if verbose: print(x.shape)
                
        return x

class CondVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        data_size: int,
        layer_size: int,
        hidden_size: int,
        
    ):
        super(CondVariationalAutoEncoder, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Linear(data_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
        )
        
        self.mu = nn.Linear(layer_size, hidden_size)
        self.logvar = nn.Linear(layer_size, hidden_size)
        
        self.conditional = nn.Embedding(num_classes, hidden_size)    
        
        self.decoder = nn.Sequential(
            nn.Linear(2*hidden_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
            nn.Linear(layer_size, data_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x, c):
        
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        
        c = self.conditional(c)
        
        z_c = torch.cat([z, c], dim=1)
        
        x = self.decoder(z_c)
        
        return x, mu, logvar

    def generate(self, z, c):
        c = self.conditional(c)
        z_c = torch.cat([z, c], dim=1)
        x = self.decoder(z_c)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        data_size: int,
        layer_size: int,
        hidden_size: int,
        
    ):
        super(VariationalAutoEncoder, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Linear(data_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
        )
        
        self.mu = nn.Linear(layer_size, hidden_size)
        self.logvar = nn.Linear(layer_size, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
            nn.Linear(layer_size, data_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        
        x = self.decoder(z)
        
        return x, mu, logvar

    def generate(self, z):
        x = self.decoder(z)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    