import numpy as np
import torch
from skimage.segmentation import watershed
from skimage.exposure import match_histograms

class EnhanceContrast:

    def __init__(self, intensity_cutoff = 0.9, dist_quantile = 0.75, dist_rebalance = 1.5, reduce_dim = True):

        # Save parameters
        self.intensity_cutoff = intensity_cutoff
        self.dist_quantile = dist_quantile
        self.dist_rebalance = dist_rebalance
        self.reduce_dim = reduce_dim

    def __call__(self, im_original, target = None):

        # Reduce dimensions
        im_original_numpy = im_original.cpu().numpy()
        im = im_original_numpy.mean(axis=0)

        # Select only cells of high intensity
        intensity_mask = (im > im.mean() * self.intensity_cutoff)

        # Compute a distance mask in order 
        # to select cells close to the center of the image
        W, H = im.shape
        distX = np.arange(H).reshape((1, H)).repeat(W, axis=0) / H - 1 / 2
        distY = np.arange(W).reshape((W, 1)).repeat(H, axis=-1) / W - 1 / 2
        dist = self.dist_rebalance * distX ** 2 + distY ** 2

        # Create mask for cells connected to bright cells close to the center of the image
        seed = np.zeros_like(intensity_mask, dtype=np.int64)
        seed[~intensity_mask] = 1
        seed[intensity_mask & (dist < np.quantile(dist[intensity_mask], self.dist_quantile))] = 2
        segmented_im = watershed(intensity_mask.astype(np.int64), seed, connectivity=5)

        # Match histogram to a uniform histogram
        if self.reduce_dim:
            matched_im = np.zeros_like(im)
            matched_im[segmented_im == 2] = match_histograms(
                im[segmented_im == 2], 
                np.linspace(0, 1, num=1024)
            )

        else:
            matched_im = np.zeros_like(im_original_numpy)
            mask = np.expand_dims(segmented_im == 2, axis=0).repeat(im_original_numpy.shape[0], axis=0)
            matched_im[mask] = match_histograms(
                im_original_numpy[mask], 
                np.linspace(0, 1, num=1024)
            )

        # Return torch tensor
        enhanced_im = torch.Tensor(matched_im).to(dtype=im_original.dtype, device=im_original.device)

        if target is None:
            return enhanced_im
        else:
            return enhanced_im, target

