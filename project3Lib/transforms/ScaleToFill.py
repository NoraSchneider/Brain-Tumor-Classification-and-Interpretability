import numpy as np
import torch
from torchvision.transforms.functional import affine

class ScaleToFill:

    # Construct helpers for getting points
    def __call__(self, im, target = None):
        
        # Helpers
        W, H = im.shape
        distX = torch.arange(H).view((1, H)).expand(W, H).to(device=im.device)
        distY = torch.arange(W).view((W, 1)).expand(W, H).to(device=im.device)

        # Compute enclosing box
        mask = im > im.min()
        upper = distY[mask].max()
        lower = distY[mask].min()
        left = distX[mask].min()
        right = distX[mask].max()

        # Transform immage by translating and scaling it
        offset = (
            (W - left - right) / 2,
            (H - upper - lower) / 2
        )
        scale = H / (upper - lower)
        
        scaled_im = affine(im.unsqueeze(0), angle=0, translate=offset, scale=scale, shear=0, fill=torch.min(im).item()).squeeze(0)
        
        if target is None:
            return scaled_im
        
        else:
            scaled_target = affine(target.unsqueeze(0), angle=0, translate=offset, scale=scale, shear=0, fill=torch.min(im).item()).squeeze(0)
            return scaled_im, scaled_target
