""" Full assembly of the parts to form the complete network """

from torch import nn
from torch import optim
from tqdm.auto import tqdm
import torch.nn.functional as F

from .utils import *
from ..utils import dice_loss, dice_coeff

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)

    
    def train_supervised(self, train_dataset, val_dataset, epochs=10, alpha = 1.0, lr = 1e-5):

        optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        # Loss function
        criterion = nn.BCELoss()
        loss_f = lambda pred, target: alpha * criterion(pred.flatten(), target.flatten()) + \
                                      (1 - alpha) * dice_loss(pred, target.unsqueeze(0), multiclass=False)

        loop = tqdm(range(epochs))
        for _ in loop:
            for x, target, _ in train_dataset:
                loss = loss_f(self(x), target)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            val_loss = self.validate_step(loop, val_dataset, loss_f)

            scheduler.step(val_loss)

    
    def train_semisupervised(self, train_dataset, val_dataset, unlabeled_dataset, epochs=10, alpha = 1.0, lr = 1e-5, beta = 1.0, sigma=0.05):

        optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        # Supervised loss is weighted average of CE between predicatons and target as well as dice score
        criterion = nn.BCELoss()
        supervised_loss_function = lambda pred, target: alpha * criterion(pred.flatten(), target.flatten()) + \
                                                        (1 - alpha) * dice_loss(pred, target.unsqueeze(0))

        # Unsupervised loss is CE of predictions
        # Scale so that only beta regulate weighting between supervised and unsupervised loss
        scale = beta * len(train_dataset) / len(unlabeled_dataset)
        unlabeled_loss_function  = lambda pred_noise, pred_base: scale * F.cross_entropy(
              torch.stack([pred_noise.flatten(), 1 - pred_noise.flatten()], dim=1), 
              torch.stack([pred_base.flatten(), 1 - pred_base.flatten()], dim=1)
          )

        loop = tqdm(range(epochs))
        for _ in loop:
            for i, (x, target, _) in enumerate(train_dataset):

                # Use one supervised datapoint
                loss = supervised_loss_function(self(x), target)

            
                # Then iterate over unsupervised examples 
                for x_unlabeld, _, _ in list(unlabeled_dataset)[i::len(train_dataset)]:

                    # Base prediction is unperterbed
                    pred_base = self(x_unlabeld)

                    # Then induce gaussian noise
                    noise = torch.empty_like(x_unlabeld)
                    noise.normal_(0, sigma).mean()
                    x_noisy = (x_unlabeld + noise).clip(min=0, max=1)
                    pred_noise = self(x_noisy)

                    # Compute consistency loss
                    loss += unlabeled_loss_function(pred_noise, pred_base)
                
                # Then optimize
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            val_loss = self.validate_step(loop, val_dataset, supervised_loss_function)

            scheduler.step(val_loss)

    def validate_step(self, loop, val_dataset, loss_function):
        
        # Validation Loop
        with torch.no_grad():
            val_loss, val_dice = 0, 0
            for x, target, _ in val_dataset:
                pred = self(x)
                val_loss += loss_function(pred, target)
                val_dice += 1 - dice_loss(pred, target.unsqueeze(0))

            loop.set_description("Loss : {:.2f} | Dice : {:.2f}".format(val_loss.item() / len(val_dataset), val_dice.item() / len(val_dataset)))
        
        return val_loss
    