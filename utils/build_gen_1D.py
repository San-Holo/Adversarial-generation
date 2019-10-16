import numpy as np
import pandas as pd

import itertools as it
import torch.optim as optim

from .build_networks_utils_1D import *

class Generator_1D(nn.Module):
    def __init__(self, deconv_sizes, latent_shape, scenario_shape, normalize, layer_norm, spectral_norm, batch_norm_proj):
        """Constructor of DCGAN_like 1D generators

        Parameters
        ----------
        deconv_sizes : int tuple list
            Each tuple describe a "deconvolution" block containing patterns of ConvTranspose1D, followed by a norm layer
            and an actvation one. We don't need to take care of the first and last layer, since they're created in the function.
            It must be shaped like this (nb_in_channel, nb_blocks, kernel_size, stride, padding) and each block is automatically
            connected to the next one in terms of size and filters.
            ! Caution !
            This constrcutor is meant for DCGAN-like architectures in 1D, so if you want to use nb_blocks > 1 in a tuple, you
            need to use proper stride and padding values. If not, you might encounter errors, because the values you're probably
            using are based on several papers in which each block contains one pattern and not several identical. By that, we mean
            that a tuple with nb_blocks > 1 will repeat a pattern with the same parameters as described in the tuple.
            For example, list = [(64, 2, 3, 1, 1), ...] would give such layers:
            ConvTranspose1D(latent_shape, 64, 3, 2, 1) -> Normalization(64) -> Activation -> ConvTranspose1D(64, 64, 3, 2, 1)
                                                                                        -> Normalization(64) -> Activation ...
        latent_shape : int
            Size of the vector sampled from the latent space

        scenario_shape : tensor size object
            Shape of the scenarios (time series) we want to generate. It will be used mainly for the last layer

        wasserstein : Boolean
            Whether we use a LayerNorm even if it's not necessary or not. It's named after the wasserstein GAn, because the original paper was proposing it just for
            the discriminator and it would have been interesting to see the effect of an other norm layer on the generator
        """
        super(Generator_1D, self).__init__()
        deconv_sizes = [(latent_shape,)] + deconv_sizes + [(scenario_shape[0], 1, 5, 3, 1)] # We add the first and the last layer here
        layers = it.chain(*[deconv_block_1D(prev[0], curr[0], curr[1], curr[2], curr[3], curr[4], scenario_shape[0], normalize=normalize, layer_norm=layer_norm,
                                         spectral_norm=spectral_norm, batch_norm_proj=batch_norm_proj) for prev, curr in zip(deconv_sizes, deconv_sizes[1:])])
        self.conv = nn.Sequential(*layers)
    def forward(self, z):
        """Forward pass with the latent vector and returns a generated image (must be viewed in another shape if you want to see it as an actual image)"""
        img = z.view(z.size(0), z.size(1), 1)
        img = self.conv(img)
        return img

    def weights_init(self):
        """Weights init in all the generator net"""
        self.apply(weights_init_1D)

