

from .resnetUnet import ResNetUNet

'''
Model ResNetUnet taken from:
https://github.com/usuyama/pytorch-unet
'''

def get_network_from_options(options):
    """ Gets the network given the options
    """
    return ResNetUNet(n_channel_in=options.n_spatial_classes, n_class_out=options.n_spatial_classes)