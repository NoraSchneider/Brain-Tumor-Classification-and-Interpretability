import torch
from torch import nn


class MaxClassifier(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = None

    def fit(self, train_dataset):

        thresholds = torch.arange(0.5, 1, 0.025)
        scores = torch.zeros_like(thresholds)

        for i in range(torch.numel(thresholds)):
            self.threshold = thresholds[i].item()

            for x, _, label in train_dataset:
                pred = self(x)
                scores[i] += (pred == label).item()

        self.threshold = torch.min(thresholds[scores == torch.topk(scores.unique(), 2)[0].min()]).item()
    
    def forward(self, x):
        pred = self.model(x)
        mask = (pred > self.threshold).view((pred.shape[0], -1))
        return torch.any(mask, dim=1)
