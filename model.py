import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    
    def __init__(
        self,
        num_channels: int,
        img_side: int,
        hidden_size: int,
    ):
        super(AutoEncoder, self).__init__()
        
        self.num_channels = num_channels
        self.img_side = img_side
        self.data_size = num_channels * img_side * img_side
        
        self.encoder = nn.Linear(self.data_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, self.data_size)
        
    def forward(self, x):
        # X is (B, C, H, W)
        
        x = x.view(-1, self.data_size)
        
        x = self.encoder(x)
        x = F.relu(x)
        x = self.decoder(x)
        x = F.sigmoid(x)
        
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        return x
    
    def encode(self, x):
        x = x.view(-1, self.data_size)
        return self.encoder(x)
    
    def decode(self, x):
        x = self.decoder(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        return x
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self, num_channels=1, dropout=0.2):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_channels, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.Sigmoid()
        )
            
        

    def forward(self, x, verbose=False):    

        x = self.encoder(x)
        if verbose: print(x.shape)
        x = self.decoder(x)
                
        return x
    
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        img_side: int,
        latent_channels: int = 4,
        dropout: float = 0.1
    ):
        
        super(VariationalAutoEncoder, self).__init__()
        
        self.img_side = img_side
        self.num_channels = num_channels
        self.latent_channels = latent_channels
        self.latent_dim = latent_channels * (img_side // 4) * (img_side // 4)
        # C_l, H/4, W/4 
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, self.latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.latent_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.mu_gen = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_gen = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 32,
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.num_channels,
                      kernel_size=1,
                      padding="same"),
            nn.Sigmoid()
        )
         
    def forward(self, x, verbose=False):
        x = self.encoder(x)
        x = x.view(-1, self.latent_dim)
        
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if verbose: print(z.shape)
        
        z = z.view(-1, 4, (self.img_side // 4), (self.img_side // 4)) # reshape to (B, C, H, W)
        
        x = self.decoder(z)
        
        return x, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.latent_dim)
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)
        return mu, logvar
    
    def decode(self, z):
        z = z.view(-1, 4, (self.img_side // 4), (self.img_side // 4))
        x = self.decoder(z)
        return x