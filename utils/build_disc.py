import numpy as np
import pandas as pd

import itertools as it
import torch.optim as optim

from .build_network_utils_2D import *

class Discriminator(nn.Module):
    def __init__(self, conv_sizes, img_shape, normalize, layer_norm, spectral_norm, wasserstein, dropout):
        """Constructor of DCGAN_like 2D discriminators

        Parameters
        ----------
        conv_sizes : int tuple list
            Each tuple describe a "convolution" block containing patterns of Conv2D, followed by a norm layer and an actvation one.
            We don't need to take care of the first and last layer, since they're created in the function.
            It must be shaped like this (nb_in_channel, nb_blocks, kernel_size, stride, padding) and each block is automatically
            connected to the next one in terms of size and filters.
            ! Caution !
            This constrcutor is meant for DCGAN-like architectures in 2D, so if you want to use nb_blocks > 1 in a tuple, you
            need to use proper stride and padding values. If not, you might encounter errors, because the values you're probably
            using are based on several papers in which each block contains one pattern and not several identical. By that, we mean
            that a tuple with nb_blocks > 1 will repeat a pattern with the same parameters as described in the tuple.
            For example, list = [(64, 2, 3, 1, 1), ...] would give such layers:
            Conv2D(latent_shape, 64, 3, 2, 1) -> Normalization(64) -> Activation -> Conv2D(64, 64, 3, 2, 1) -> Normalization(64) -> Activation ...
            
        img_shape : tensor size object
            Shape of the images taken in input. It will be used mainly for the first layer

        wasserstein : Boolean
            Whether we remove the sigmoid or not. It's named after the wasserstein GAN, because the original framework does not optimize the same way
            as a classic GAN.

        gradient_penalty : Boolean
            Whether we use a LayerNorm or not, in every conv_block. It's named after the wasserstein GAN with gradient penalty, because the original paper
            advices it for the discriminator

        dropout : (Boolean, float) tuple
            Whether we use dropout or not, in every conv_block after the activations. It's related to the WGAN-GP-CT, where it was used.
        """
        super(Discriminator, self).__init__()
        conv_sizes = [(img_shape[0],)] + conv_sizes + [(1, 1, 4, 1, 0)]
        layers = it.chain(*[conv_block(prev[0], curr[0], curr[1], curr[2], curr[3], curr[4], 1, normalize=normalize, wasserstein=wasserstein,
                                       layer_norm=layer_norm, spectral_norm=spectral_norm, dropout=dropout) for prev, curr in zip(conv_sizes, conv_sizes[1:])])
        self.conv = nn.Sequential(*layers)
        self.tmp_layers =(self.conv[:-1], self.conv[-1]) # In order to use torchsummary efficiently
        
    def forward(self, img, with_pre_last=False):
        """Forward pass with an image, and return result of forward without and with the last layer if without_last is True (WGAN-GP-CT related)"""
        if with_pre_last:
            output_without_last = self.tmp_layers[0](img)
            output = self.tmp_layers[1](output_without_last)
            return output, output_without_last
        else:
            output = self.conv(img)
            return output.view(-1)

    def weights_init(self):
        """Weights init in all the discriminator net"""
        self.apply(weights_init)



