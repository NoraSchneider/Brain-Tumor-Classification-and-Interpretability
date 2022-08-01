from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
import shap
from torch import nn


"""
    Function for interpreting the UNet model with GradCam
"""
def gradcam_unet(base_model, layer, sample, target, label):

    mask = (target  == label) & (sample[0] > sample.min())

    def helper(x):
        pred = base_model(x)[:, mask].mean(dim=1).unsqueeze(-1)
        return pred if label == 1 else 1 - pred

    sample.requires_grad = True
    layer_gc = LayerGradCam(helper, layer)
    attributions_1 = layer_gc.attribute(sample, 0)
    return LayerAttribution.interpolate(attributions_1, (128, 128))


"""
    Function for interpreting the UNet model with Integrated Gradients
"""
def integrad_unet(base_model, sample, target, label):
    
    mask = (target  == label) & (sample[0] > sample.min())

    def helper(x):
        pred = base_model(x)[:, mask].mean(dim=1).unsqueeze(-1)
        return pred if label == 1 else 1 - pred
    
    sample.requires_grad = True
    ig = IntegratedGradients(helper)
    return ig.attribute(sample, 0)


"""
    Wrapper class for the UNet model when interpreting it with shap
"""
class HelperShap(nn.Module):

        def __init__(self, model, mask):
            super().__init__()
            self.model = model
            self.mask = mask

        def __call__(self, x):
            return self.model(x)[:, self.mask].mean().unsqueeze(0).unsqueeze(0)


"""
    Function for interpreting the UNet model with Integrated Gradients
"""
def shap_unet(base_model, background, sample, target, label):
    mask = (target  == label) & (sample[0] > sample.min())
    e = shap.DeepExplainer(HelperShap(base_model, mask), background)
    return e.shap_values(sample)
