import torch
import torch.nn as nn
import torch.nn.functional as F

class PCA(nn.Module):
    """
    A PCA is a simple autoencoder with a single linear layer in the encoder and decoder
    No activation functions are used
    
    Note: in a PCA, the decoder is the same as the encoder
    """
    def __init__(self,
                 num_channels: int,
                    img_side: int,
                    hidden_size: int):
        super(PCA, self).__init__()
        input_size = num_channels * img_side * img_side
        
        self.matrix = nn.Parameter(torch.randn(input_size, hidden_size))
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = torch.matmul(x, self.matrix)
        x = torch.matmul(x, self.matrix.t())
        x = F.sigmoid(x) 
        x = x.view(B, C, H, W)

        return x

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
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, self.data_size),
            nn.Unflatten(1, (num_channels, img_side, img_side))
        )
        
    def forward(self, x):
        # X is (B, C, H, W)
        
        x = x.view(-1, self.data_size)
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x
    
    def encode(self, x):
        x = x.view(-1, self.data_size)
        return self.encoder(x)
    
    def decode(self, x):
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self, num_channels=1, dropout=0.5):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, num_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
            
    def get_shape(self, x): 
        x = self.encoder(x)
        print(torch.prod(torch.tensor(x.shape)))
        return x.shape

    def forward(self, x, verbose=False):    

        x = self.encoder(x)
        if verbose: print(x.shape)
        x = self.decoder(x)
                
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", dropout=0.5):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))
        x = self.dropout(x)
        
        x = self.pool(x)
    
        return x
    
class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=(1,1), padding=1):
        super(ConvTBlock, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides[0], padding=1) # always same padding
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.convt2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=strides[1], padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.convt1(x)
        x = F.leaky_relu(self.bn1(x))
        
        x = self.convt2(x)
        x = F.leaky_relu(self.bn2(x))
        
        return x
        
class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        img_side: int,
        latent_dim: int,
        dropout: float = 0.5
    ):
        
        super(VariationalAutoEncoder, self).__init__()
        
        self.img_side = img_side
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        
        
        self.encoder = nn.Sequential(
            ConvBlock(num_channels, 64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128, 256, dropout=dropout),
            nn.Flatten(),
            nn.Linear(256 * (img_side // 8) * (img_side // 8), self.latent_dim),
            nn.LeakyReLU()
        )
        
        self.mu_gen = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_gen = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*10),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim*10, 256 * (img_side // 8) * (img_side // 8)),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, img_side // 8, img_side // 8)),
            
            ConvTBlock(256, 256, strides =(1,1)),
            ConvTBlock(256, 128, strides =(1,2), padding=0),
            nn.AdaptiveMaxPool2d((img_side//4, img_side//4)),
            
            ConvTBlock(128, 128, strides =(1,1)),
            ConvTBlock(128, 64, strides =(1,2), padding=0),
            nn.AdaptiveMaxPool2d((img_side//2, img_side//2)),

            ConvTBlock(64, 64, strides =(1,1)),
            ConvTBlock(64, 32, strides =(1,2), padding=0),
            nn.AdaptiveMaxPool2d((img_side, img_side)),
            
            nn.Conv2d(32, num_channels, kernel_size=1, stride=1, padding=0), # adjust to num_channels

            nn.Sigmoid()
        )
        
        print("Latent dim:", self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
            
        x = self.decoder(z)
    
        
        return x, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
if __name__ == '__main__':
    model = ConvAutoencoder(num_channels=3)
    x = torch.randn(5, 3, 32, 32)
    print(x.shape)
    
    output = model(x)
    
    print(output.shape)
    
    
