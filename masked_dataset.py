from opcode import hasconst
import numpy as np
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder

from PIL import Image

from typing import Tuple, Any

class MaskedDataset(ImageFolder):

    def __init__(self, mask_folder : Path, device = "cpu", common_transforms = [], use_empty_mask = False,  *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mask_folder = mask_folder
        self.device = device
        self.common_transforms = common_transforms
        self.use_empty_mask = use_empty_mask

        for t in common_transforms:
            if hasattr(t, "fit"):
                t.fit(self)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = Path(self.samples[index][0])
        sample = self.loader(path)
        target, label = self.load_mask(path, sample)

        
        if self.transform is not None:
            sample = self.transform(sample).to(device=self.device)
            target = self.transform(target).to(device=self.device)

        for t in self.common_transforms:

            # Helper for fitting transforms
            if hasattr(t, "fit") and not t.fitted:
                break

            # Transform in paralell
            sample, target = t(sample, target)

        target = (target[0] > 0).to(dtype=sample.dtype)

        # Ensure that shapes are correct
        # Sample: [<Batch>, <Channel>, <Width>, <Height>]
        # Target: [<Batch>, <Width>, <Height>]
        sample = sample.view((1,) * (4 - len(sample.shape)) + sample.shape)
        target = target.view((1,) * (3 - len(target.shape)) + target.shape)


        return sample, target, label

    def load_mask(self, path : Path, sample):

        mask = np.zeros_like(np.array(sample))

        folder = str(path.parent.stem)
        if folder == "yes" and not self.use_empty_mask:
            mask = np.load(self.mask_folder.joinpath(path.stem + ".npy")).astype(np.uint8)
            target = Image.fromarray(np.repeat(mask[:, :, np.newaxis], 3, 2))
        
        else:
            target = Image.fromarray(np.zeros_like(np.array(sample)))

        return target, int(folder == "yes")
