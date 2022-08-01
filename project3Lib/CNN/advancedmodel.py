import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)
        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        images, labels = batch
        # Get predictions
        out = self(images)
        # Get loss
        loss = F.cross_entropy(out, labels)
        # Get accuracy
        _, preds = torch.max(out, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        return {'val_loss': loss, 'val_acc': acc}