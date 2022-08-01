from genericpath import exists
from pathlib import Path
import shutil
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def make_uniqe_dataset(verbose = True, base_folder = Path("data/images"), unique_folder = Path("data/unique_images")):
    # Define data transform
    train_transform = transforms.Compose([
            transforms.Resize(128),             # resize shortest side to 128 pixels
            transforms.CenterCrop(128),         # crop longest side to 128 pixels at center
            transforms.ToTensor()               # convert PIL image to tensor
    ])
    
    for sub_folder in base_folder.glob("./*"):

        # Initialize train/test sets
        data_paths = list(sub_folder.glob("./*"))
        train_dataset = [train_transform(Image.open(open(path, "rb")).convert("RGB")) for path in data_paths]

        unique_images = []
        for i in range(len(train_dataset)):

            no_collisions = True
            
            # Test if matches any image up until now
            for j in range(i):
                if torch.norm(train_dataset[i][0] - train_dataset[j][0]) < 1:

                    no_collisions = False

                    # Plot and print difference in norm
                    if verbose:
                        f, axarr = plt.subplots(1, 2)
                        axarr[0].imshow(train_dataset[i][0], cmap='gray')
                        axarr[1].imshow(train_dataset[i][0], cmap='gray')
                        plt.show()
                        print(i, j, torch.norm(train_dataset[i][0] - train_dataset[j][0]))
                    
                    break
            
            # If no repition append to unique set
            if no_collisions:
                unique_images.append(i)

        # Make new folder for unique images
        unique_subfolder = unique_folder.joinpath(sub_folder.stem)
        unique_subfolder.mkdir(parents=True, exist_ok=True)

        # Copy to that folder
        for i in unique_images:
            shutil.copy(data_paths[i], unique_subfolder.joinpath(data_paths[i].stem))


if __name__ == "__main__":
    make_uniqe_dataset()
