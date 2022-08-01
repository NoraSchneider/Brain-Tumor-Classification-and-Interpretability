from sklearn.decomposition import PCA
import numpy as np
import torch 
from skimage.segmentation import slic

import matplotlib.pyplot as plt

class SubtractPCA:

    def __init__(self, n_components = 10):
        self.pca = PCA(n_components = n_components)
        self.fitted = False

    def fit(self, dataset):

        X = np.array([x.flatten().cpu().numpy() for x, _, label in dataset if label == 0])
        self.pca.fit(X)
        self.fitted = True

    def __call__(self, im, target = None):
        
        # Have a local copy on cpu
        im_numpy = im.cpu().numpy().reshape(1, -1)

        # Compress image
        im_compressed = self.pca.inverse_transform(self.pca.transform(im_numpy))
        im_compressed[im_numpy == im_numpy.min()] = 0

        # Return residual and keep target unchanged
        residual = np.abs(im_numpy - im_compressed)
        residual = torch.from_numpy(residual).to(device=im.device, dtype=im.dtype).view(im.shape)

        return residual, target
