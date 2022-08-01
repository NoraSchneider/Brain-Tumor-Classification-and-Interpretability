import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop, lr_scheduler
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ------ ENCODER MODULE

class VariationalEncoder(nn.Module):
    def __init__(self, imgChannels=1, f_dim=1024, z_dim=540):
        
        super(VariationalEncoder, self).__init__()
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 32, kernel_size=3, stride=2)
        self.encConv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.encConv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.encConv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.encConv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.encConv6 = nn.Conv2d(128, 256, kernel_size=3)
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.fc1 = nn.Linear(f_dim, z_dim)
        self.fc2 = nn.Linear(f_dim, z_dim)

        self.N = torch.distributions.Normal(0, 1)

        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda() 
            self.N.scale = self.N.scale.cuda()
        
        self.temp_shape = 0
        
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        
        x = F.leaky_relu(self.encConv1(x))
        x = F.leaky_relu(self.encConv2(x))
        x = F.leaky_relu(self.encConv3(x))  
        x = F.leaky_relu(self.encConv4(x))
        x = F.leaky_relu(self.encConv5(x))
        x = F.leaky_relu(self.encConv6(x))
             
        # FLATTEN
        self.temp_shape = x.shape
        #print(self.temp_shape)                               
        x = self.flatten(x)                         
        #print(x.shape)
        
        #GET MEAN AND VAR
        mu =  self.fc1(x)
        sigma = torch.exp(self.fc2(x))
        
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        #print("after kl")
        #print(z.shape)
        return z




# ------ DECODER MODULE

class Decoder(nn.Module):
    def __init__(self, imgChannels=3,  f_dim=1024, z_dim=540):
        
        super(Decoder, self).__init__()
    
        self.fc3 = nn.Linear(z_dim, f_dim)
        self.decConv1 = nn.ConvTranspose2d(f_dim, 256, kernel_size=3)
        self.decConv2 = nn.ConvTranspose2d(256, 128, kernel_size=3)
        self.decConv3 = nn.ConvTranspose2d(128, 128, kernel_size=3)
        self.decConv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.decConv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.decConv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.decConv7 = nn.ConvTranspose2d(32, imgChannels, kernel_size=4, stride=2)
        self.f_dim = f_dim
        
        
    def forward(self, z):
        x =  self.fc3(z)
        x = x.view(x.size(0), self.f_dim, 1, 1)
        x = F.leaky_relu(self.decConv1(x))
        x = F.leaky_relu(self.decConv2(x))
        x = F.leaky_relu(self.decConv3(x))
        x = F.leaky_relu(self.decConv4(x))
        x = F.leaky_relu(self.decConv5(x))
        x = F.leaky_relu(self.decConv6(x))
        x = torch.sigmoid(self.decConv7(x))
        #print(x.shape)
        return x



# ------ Variational Autoencoder

class VariationalAutoencoder(nn.Module):
    def __init__(self, imgChannels=3, f_dim=2304, z_dim=540):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(imgChannels=imgChannels, f_dim=f_dim, z_dim=z_dim)
        self.decoder = Decoder(imgChannels=imgChannels, f_dim=f_dim, z_dim=z_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


    def train(self, data, epochs=20,  lr=0.0001, save=""):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            train_loss = []
            for x, y in data:
                x = x.to(device) # GPU
                opt.zero_grad()
                x_hat = self(x)
                loss = ((x - x_hat)**2).sum() + self.encoder.kl
                loss.backward()
                opt.step()
            
                train_loss.append(loss.data.to('cpu').detach().numpy())
        
            to_print = "Epoch[{}/{}] Avg Loss: {:.3f}".format(epoch+1, 
                                    epochs, np.mean(train_loss))
            print(to_print)

        if save:
            torch.save(self.state_dict(), save)
            
        return self



def plot_latent(autoencoder, data, num_batches=20):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break




def plot_reconstructed(autoencoder, dataloader, r0=(-10, 10), r1=(-10, 10), dims=(0,1), n=15, nChannels=3):
    w = 128
    img = np.zeros((n*w, n*w))
    
    start_embed = autoencoder.encoder( next(iter(dataloader))[0][0].unsqueeze(0).to(device) ).detach()
    
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):

            new_embed = start_embed.to('cpu').clone().to(device)
            
            new_embed[:,dims[0]] = start_embed[:,dims[0]] + y
            new_embed[:,dims[1]] = start_embed[:,dims[1]] + x
            z = new_embed
            x_hat = autoencoder.decoder(z)
            if nChannels > 1:
                x_hat = x_hat.reshape(nChannels,128, 128).to('cpu').detach().numpy()[0]
            else:
                x_hat = x_hat.to('cpu').detach().numpy().squeeze()
                
            img[i*w:(i+1)*w, j*w:(j+1)*w] = x_hat
    plt.figure(figsize = (10,10))
    plt.imshow(img, extent=[*r0, *r1])