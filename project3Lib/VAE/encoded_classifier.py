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

from .vae import *

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Encoder_Classifier(nn.Module):
    def __init__(self,  vae_path='vae_trans_3.torch', imgChannels=3, vae_trainable = True, 
                 z_dim=540):
        
        super(Encoder_Classifier, self).__init__()
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder

        vae_loaded  = VariationalAutoencoder(imgChannels=imgChannels)
        vae_loaded.load_state_dict(torch.load(vae_path))
        
        self.encoder = vae_loaded.encoder

        for param in self.encoder.parameters():
            param.requires_grad = vae_trainable
            
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 2)

    def forward(self, x):
        
        encoded = self.encoder(x)
        
        y =  self.fc1(encoded)
        y =  self.fc2(y)
        y =  self.fc3(y)
        y =  self.fc4(y)
        y =  self.fc5(y)
        y = torch.softmax(self.fc6(y), dim=-1)

        return y
    

    def training_step(self, batch):
        images, labels = batch 

        out = self(images.to(device))         
        # Get loss        
        loss = F.cross_entropy(out, labels.to(device)) 
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch 
        # Get predictions
        out = self(images.to(device))                    
        # Get loss
        loss = F.cross_entropy(out, labels.to(device))   
        # Get accuracy
        _, preds = torch.max(out, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())          
        return {'val_loss': loss, 'val_acc': acc}


    def validation(self, valid_loader):
    
        self.eval()
        loss_total = 0
        acc_total = 0

        # Test validation data
        with torch.no_grad():
            for data in valid_loader:
                input = data[0].to(device)
                label = data[1].to(device)
            
                res = self.validation_step(data)
                loss = res['val_loss']
                loss_total += loss

                acc = res['val_acc']
                acc_total += acc
            
        return loss_total / len(valid_loader), acc_total/ len(valid_loader)

    # Train
    def traindata(self, train_loader, valid_loader, epochs=20,
                    patience=10, lr=0.000001, save="", print_statements = True):
        # Early stopping
        last_loss = 200
        patience = patience
        triggertimes = 0

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

    
        for epoch in tqdm.tqdm(range(1, epochs+1)):
            self.train()

            for times, data in enumerate(train_loader, 1):
    
                # Zero the gradients
                optimizer.zero_grad()
    
                # Forward and backward propagation
                loss = self.training_step(data)
                loss.backward()
                optimizer.step()

                # Show progress
                if times % 100 == 0 or times == len(train_loader):
                    if(print_statements): print('[{}/{}, {}/{}] total loss: {:.8} | '.format(epoch, epochs, times, len(train_loader), loss.item()))

            # Early stopping
            current_loss, current_acc = self.validation(valid_loader)
            if(print_statements): print('The Current Loss:', current_loss, "The Current Acc:", current_acc)

            if current_loss > last_loss:
                trigger_times += 1
                if(print_statements): print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    if(print_statements): print('Early stopping!\nStart to test process.')
                    if save:
                        torch.save(self.state_dict(), save)
                    return best_model

            else:
                if(print_statements): print('trigger times: 0')
                trigger_times = 0
                best_model = self

            last_loss = current_loss


        if save:
                torch.save(self.state_dict(), save)
        
        return self


    def test(self, testloader):
        loss_total = 0
        acc_total = 0

        # Test validation data
        with torch.no_grad():
            self.eval()
            for data in testloader:
                input = data[0].to(device)
                label = data[1].to(device)

                res = self.validation_step(data)
                loss = res['val_loss']
                loss_total += loss

                acc = res['val_acc']
                acc_total += acc
        acc = acc_total/len(testloader)
        loss = loss_total/len(testloader)
    
        return acc, loss

    def predict(self, x):
        out = self(x.unsqueeze(0))   
        _,prediction = torch.max(out, dim=1)
        return prediction[0].item(), out