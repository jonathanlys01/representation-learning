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

        # dropout can be applied to all layers
        self.dropout = nn.Dropout(dropout)
        
        # pooling for the first ones (downsampling)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding=1)  
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        

        self.t_conv1 = nn.ConvTranspose2d(4, 16, 5, dilation=2) 
        self.t_bn1 = nn.BatchNorm2d(16)
        self.t_conv2 = nn.ConvTranspose2d(16, 16, 5, dilation=2)
        self.t_bn2 = nn.BatchNorm2d(16)
        self.t_conv3 = nn.ConvTranspose2d(16, num_channels, 5, dilation=2)
        

    def forward(self, x, verbose=False):    

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.t_conv1(x))
        x = self.t_bn1(x)
        x = F.relu(self.t_conv2(x))
        x = self.t_bn2(x)
        x = F.sigmoid(self.t_conv3(x))
        
                
        return x

class CondVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        
        num_channels: int,
        img_side: int,
        
        layer_size: int,
        hidden_size: int,
        
    ):
        super(CondVariationalAutoEncoder, self).__init__()
        
        self.num_channels = num_channels
        self.img_side = img_side
        self.data_size = num_channels * img_side * img_side
        
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, layer_size),
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
            nn.Linear(layer_size, self.data_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x, c):
        
        x = x.view(-1, self.data_size)
        
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        
        c = self.conditional(c)
        
        z_c = torch.cat([z, c], dim=1)
        
        x = self.decoder(z_c)
        
        
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        
        return x, mu, logvar

    def generate(self, z, c):
        c = self.conditional(c)
        z_c = torch.cat([z, c], dim=1)
        x = self.decoder(z_c)
        
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        return x
    
    def encode(self, x):
        x = x.view(-1, self.data_size)
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        img_side: int,
        layer_size: int,
        hidden_size: int,
        
    ):
        super(VariationalAutoEncoder, self).__init__()
        
        self.num_channels = num_channels
        self.img_side = img_side
        self.data_size = num_channels * img_side * img_side
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
        )
        
        self.mu = nn.Linear(layer_size, hidden_size)
        self.logvar = nn.Linear(layer_size, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(layer_size),
            nn.Linear(layer_size, self.data_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        
        x = x.view(-1, self.data_size)
        
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        
        x = self.decoder(z)
        
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        
        return x, mu, logvar

    def generate(self, z):
        x = self.decoder(z)
        
        x = x.view(-1, self.num_channels, self.img_side, self.img_side)
        return x
    
    def encode(self, x):
        
        x = x.view(-1, self.data_size)
        
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    