import torch
from torch import nn
import torch.utils
import torch.distributions
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import shap



from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from matplotlib import colors

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
data                      : single data point to apply function to 
model                     : model to apply function to 
idx       = 1             : idx of the datapoint, for file saving and plot title purposes
grad_type = "integ_grads" : choose between "integ_grads" and "grad_cam"
plot      = True          : if plot is True, images are plotted, if False, they are returned as arrays
layer     = None          : layer to use the function on -- alterantively the index of teh layer can be given for some models
layer_idx = 4             : idx of the layer to apply grad cam to / not default option
save_name = ""            : will save images to files if file name is given -- only a unique prefix to add to final name
"""

def plot_grads(data, model,  idx=1, grad_type= "integ_grads" ,plot=True, layer=None,layer_idx=-1,  save_name=""):

    
    if(len(data[0].shape)<=3):
        sample = data[0].unsqueeze(0)
    else:
        sample = data[0]
    #sample.requires_grad = True
    sample_class = data[1]

    model = model.to(device)
    sample = sample.to(device)
    
    if grad_type == "integ_grads":

        ig = IntegratedGradients(model)
        attributions_1, approximation_error_1 = ig.attribute(sample, target=sample_class,
                                            return_convergence_delta=True)
        attributions_2, approximation_error_2 = ig.attribute(sample, target=1-sample_class,
                                            return_convergence_delta=True)
    else: # grad cam
        
        #LayerGradCam takes model and indexed layer / given layer from model
        if layer_idx > -1:
            layer_gc = LayerGradCam(model, model.model[layer_idx])
        else:
        	layer_gc = LayerGradCam(model, layer)
        sample.requires_grad = True
        
        #attributions for correct and incorrect classes
        attributions_1 = layer_gc.attribute(sample, sample_class)
        attributions_2 = layer_gc.attribute(sample, 1-sample_class)
        
        #interpolating attributions to original image shape
        attributions_1 = LayerAttribution.interpolate(attributions_1, (128, 128))
        attributions_2 = LayerAttribution.interpolate(attributions_2, (128, 128))
        
        
    if plot:
        f, axs = plt.subplots(1,3, figsize=(20, 10))
        f.suptitle(f"Integrated Gradients for Image {idx}", fontsize=14)
        divnorm=colors.TwoSlopeNorm(vcenter=0.)

        if attributions_1.shape == (1,3,128,128):
            attributions_1 = attributions_1[:,0]
            attributions_2 = attributions_2[:,0]
            
        if sample.shape == (1,3,128,128):
            sample = sample[:,0]
                                                         
        axs[0].imshow( np.array(sample.detach().cpu().numpy().squeeze()), cmap= 'gray')
        axs[0].set_title(f'Original Image')                                               
        
        axs[1].imshow( np.array(attributions_1.detach().cpu().numpy().squeeze()),
                    cmap= 'RdBu', norm=divnorm)
        axs[1].set_title(f'Integrated Gradients \n for Class {sample_class} (Correct)')
        axs[2].imshow( np.array(attributions_2.detach().cpu().numpy().squeeze()),
                    cmap= 'RdBu', norm=divnorm)
        axs[2].set_title(f'Integrated Gradients \n for Class {1-sample_class} (Incorrect)') 

        if save_name:                                                    
            f.savefig(f'images/{grad_type}/{save_name}_{grad_type}_{idx}_{sample_class}.png')
        
    else:
        return attributions_1, attributions_2


"""
version of the function above that gets a dataset instead of a single data point

"""


def plot_grads_dataloader(dataset, model, grad_type= "integ_grads" ,plot=True, layer=None, layer_idx=-1, save_name=""):

    it = iter(dataset)

    for i in range(len(dataset)):
    
        plot_grads(data=next(it), model=model, idx=i, grad_type= grad_type, plot=plot,layer=layer, layer_idx=layer_idx, save_name=save_name)

