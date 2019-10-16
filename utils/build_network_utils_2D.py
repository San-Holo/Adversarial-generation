import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def conv_block(in_filter, output_filter, nb_conv, kernel_size, stride, padding, final_nbchannels, normalize, wasserstein, layer_norm, spectral_norm, dropout, activation_function=nn.LeakyReLU(0.2, inplace=True)):
    """To simplify the creation of convolutional sequences for discriminator

    Parameters
    ----------
    in_filter :  int
        Number of filters that we want in entry

    output_filter :  int
        Number of filters that we want in output

    nb_conv : int
        Number of convolution layers

    kernel_size, stride, padding : int
        We assume that the kernel is a square

    final_nbchannels : int
        Where to stop the classic pattern Conv, Norm, Act
        
    activation_function : nn Function
        Activation function after each convolution

    normalize : boolean
        Add normalization or not

    wasserstein : boolean
        If True, we must remove sigmoid at the end

    layer_norm : boolean
        If True, we use LayerNorm instead of batch norm -> As it's done in WGAN-GP

    spectral_norm : boolean
        If true, we use SpectralNorm instead of others -> Seems to be the real state of the art
        
    dropout : (Boolean, float) tuple
        Whether we use dropout or not, and its corresponding probability. It was used in wassertein-GAN-GP-CT paper.
        
    Returns
    ---------
    sequential : Sequential torch Object
        The convolutional sequence that we were seeking
    """
    nbchannel = in_filter
    nbfilter = output_filter
    sequential = []
    for i in range(nb_conv):
        # Had to change the code here, instead of using my own implementation
        if layer_norm and spectral_norm: # No one used both of them at the same time -> Logical
            raise ValueError
        else:
            if spectral_norm:
                tmp_conv = nn.utils.spectral_norm(nn.Conv2d(nbchannel, nbfilter, kernel_size, stride, padding, bias=False))
            else:
                tmp_conv = nn.Conv2d(nbchannel, nbfilter, kernel_size, stride, padding, bias=False)
            sequential.append(tmp_conv)
            nbchannel = nbfilter
            if nbchannel != final_nbchannels:
                if normalize:
                    if layer_norm:
                        sequential.append(nn.GroupNorm(1, nbfilter))
                    else:
                        sequential.append(nn.BatchNorm2d(nbfilter))
                sequential.append(activation_function)
                if dropout[0]:
                    sequential.append(nn.Dropout(p=dropout[1]))
            else:
                if not wasserstein:
                    sequential.append(nn.Sigmoid())
    return sequential

def deconv_block(latent_vector_size, output_channels, nb_deconv, kernel_size, stride, padding, final_nbchannels, normalize, layer_norm, spectral_norm, batch_norm_proj, activation_function=nn.ReLU()):
    """To simplify the creation of fractionned strided convolutional sequences for the generator

    Parameters
    ----------
    latent_vector_size :  int
        Dimensionnality of the sampled vector

    output_channels :  int
        Number of filters ~ channels that we want in output

    nb_deconv : int
        Number of deconvolution layers (Deconvolution is not a correct term there though)

    kernel_size, stride, padding : int
        We assume that the kernel is a square

    final_nbchannels : int
        Where to stop the classic pattern Conv, Norm, Act

    activation_function : nn Function
        Activation function after each fractionned strided convolution

    normalize : boolean
        Add normalization or not

    layer_norm : boolean
        If True, we must use LayerNorm instead of Batch norm -> As it's done in WGAN-GP

    spectral_norm : boolean
        If true, we use SpectralNorm instead of others -> Seems to be the real state of the art

    batch_norm_proj : boolean
        If True, add Batch norm right after spectral norm layer -> As described in Self-Attention GAN.
        Actually, it's quite different from SAGAN but we take it as a source of inspiration

    Returns
    ---------
    sequential : Sequential torch Object
        The convolutional sequence that we were seeking
    """
    nbchannel = latent_vector_size
    nbfilter = output_channels
    sequential = []
    for i in range(nb_deconv):
        # Had to change the code here, instead of using my own implementation
        if layer_norm and spectral_norm: # No one ever used both of them at the same time -> Logical
            raise ValueError
        else:
            if spectral_norm:
                tmp_deconv = nn.utils.spectral_norm(nn.ConvTranspose2d(nbchannel, nbfilter, kernel_size, stride, padding, bias=False))
            else:
                tmp_deconv = nn.ConvTranspose2d(nbchannel, nbfilter, kernel_size, stride, padding, bias=False)
            sequential.append(tmp_deconv)
            nbchannel = nbfilter
            if nbchannel != final_nbchannels:
                if normalize:
                    if layer_norm:
                        sequential.append(nn.GroupNorm(1, nbfilter))
                    elif spectral_norm and batch_norm_proj:
                        sequential.append(nn.BatchNorm2d(nbfilter))
                    else:
                        sequential.append(nn.BatchNorm2d(nbfilter))
                sequential.append(activation_function)
            else:
                sequential.append(nn.Tanh())
    return sequential

def network_from_shape(net_structure, activation=nn.ReLU()):
    """To simplify the creation of fully connected layers sequences

    Parameters
    ----------
    net structure: int list
        Describe each layer size -> one entry of the list is a layer conv_size

    activation_function : nn Function
        Activation function after each layer of the net

    Returns
    ---------
    temp :  Torch object list
        The fully connected sequence with the last activation function "tanh"
    """
    temp = []
    for prev, next in zip(net_structure[:-1], net_structure[1:]):
         temp.append(nn.Linear(prev, next))
         temp.append(activation)
    temp = temp[:-1] # Remove last activation
    return temp

def weights_init(module, mean=0.0, std=0.02):
    """To init weights in a layer according to the vast majority of papers

    Parameters
    ----------
    module : torch nn module
        Each module = A layer usually

    mean : float
        Mean of the normal distribution used to init a layer

    std : float
        Standard deviation of the normal distribution used to init a layer    
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.normal_(module.weight.data, mean, std)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, std)
        nn.init.constant_(module.bias.data, 0)
