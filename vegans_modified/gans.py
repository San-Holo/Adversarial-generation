# -*- coding: utf-8 -*-

# Classic libraries
import numpy as np
import pandas as pd
from random import shuffle, seed
from math import floor
from abc import ABC, abstractmethod
from functools import partialmethod
import torch
import torch.nn as nn


# Datasets loading
from torch.utils.data.dataset import Dataset
import torchvision.utils as vutils

# Vizualisation
from tqdm import tqdm
from torchsummary import summary

# Personal version of an open_source library
from .gan_gp import GANGP as gan_gp
from .gan_gp_ct import GANGP_CT as gan_gp_ct
from .wgan import WGAN as wgan
from .wgan_gp import WGANGP as wgan_gp
from .wgan_gp_ct import WGANGP_CT as wgan_gp_ct
from .utils import plot_losses, plot_image_samples

# Networks and data utils
from utils.build_gen import *
from utils.build_disc import *
from utils.build_gen_1D import *
from utils.build_disc_1D import *
from data_loaders import * # To refactor based on your tree view

# Save handlers
import os
import pickle
import json

# Add optimization
# import ax
seed(42)
torch.manual_seed(42)


# Models #
##########

class MODEL(ABC):
    def __init__(self, parser, path):
        """To init a GAN class

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters for the model

        path : String
             Where the trained generator and discriminator will be saved, and results too
        """
        self.save_dir = path
        self.dataset_size = None # May change in the future
        one_dim = parser.get('one_dim')
        resnet = parser.get('resnet')
        # Will be used as it's given
        self.epoch = parser.get('epochs')
        # Will be optimized
        self.batch_size = parser.get('batch_size')
        self.latent_size = parser.get('latent_size')
        self.lr_g = parser.get('lr_g')
        self.lr_d = parser.get('lr_d')
        self.B1 = parser.get('beta1')
        self.B2 = parser.get('beta2')
        self.d_iters = parser.get('d_iters')
        self.long_d_iters = parser.get('long_d_iters')
        # Setup parameters
        self.mode = parser.get('data')
        self.format = parser.get('format')
        self.model_name = parser.get('type')
        self.conv_sizes_gen = parser.get('conv_sizes_gen')
        self.conv_sizes_dis = parser.get('conv_sizes_dis')
        self.normalize_disc = parser.get('normalize_disc')
        self.wasserstein = parser.get('wasserstein')
        self.layer_norm_disc = parser.get('layer_norm_disc')
        self.spectral_norm_disc = parser.get('spectral_norm_disc')
        self.dropout = (parser.get('dropout'), parser.get('dropout_value'))
        self.normalize_gen = parser.get('normalize_gen')
        self.layer_norm_gen = parser.get('layer_norm_gen')
        self.spectral_norm_gen = parser.get('spectral_norm_gen')
        self.batch_norm_proj_gen = parser.get('batch_norm_proj_gen')
        # Dataloader structures
        if one_dim:
            dataset = SeriesAsDataset()
            scenario_shape = dataset.scenarios_size
        else:
            dataset = SeriesAsImagesDataset(method=self.mode, ext=self.format)
            scenario_shape = dataset.scenarios_size
        self.train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        # Model's history
        self.hist_train = {}
        self.hist_train['loss_d_batch'] = []
        self.hist_train['loss_g_batch'] = []
        self.hist_train['loss_d_epoch'] = []
        self.hist_train['loss_g_epoch'] = []     
        self.hist_train['fixed_noises'] = []
        # Building model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_params = (self.conv_sizes_gen, self.latent_size, scenario_shape, self.normalize_gen, self.layer_norm_gen, self.spectral_norm_gen, self.batch_norm_proj_gen)
        disc_params = (self.conv_sizes_dis, scenario_shape, self.normalize_disc, self.layer_norm_disc, self.spectral_norm_disc, self.wasserstein, self.dropout)
        self.generator, self.discriminator = self.GenAndDisc(one_dim, resnet, self.device, gen_params, disc_params)
        # Optimizers
        self.goptimizer = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.B1, self.B2))
        self.doptimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(self.B1, self.B2))
        self.fixed_noise = torch.randn(scenario_shape[1], self.latent_size, device=self.device)
        summary(self.generator, (self.latent_size,))
        if one_dim:
            summary(self.discriminator, (scenario_shape[0], scenario_shape[1]))
        else:
            summary(self.discriminator, (scenario_shape[0], scenario_shape[1], scenario_shape[2]))
        
    @abstractmethod
    def train(self):
        """Training function for GAN objects -> No parameters"""
        print("====== TASK DESCRIPTION =======")
        print("Model : ", self.model_name)
        print("Type of images : ", self.mode)
        print("Batch size : ", self.batch_size)
        print("Save dir : ", self.save_dir)
        print("=" * 31)
        pass

    def save(self, parser):
        """Saving function for GAN objects -> No parameters"""
        save_dir = os.path.join(self.save_dir, self.mode, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.generator.state_dict(), os.path.join(save_dir, self.model_name + '_G.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, self.model_name + '_D.pth'))
        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.hist_train, f)
        with open(os.path.join(save_dir, self.model_name + '_hyperparameters.pkl'), 'w') as f_:  
            json.dump(parser, f_)

    # Quick utils #
    ###############
    def GenAndDisc(self, one_dim, resnet, device, args_gen, args_disc):
        if one_dim:
            self.model_name += '_conv1D'
            gen = Generator_1D(*args_gen)
            disc = Discriminator_1D(*args_disc)
        elif resnet:
            self.model_name += '_resnet'
            pass
        elif resnet and one_dim:
            self.model_name += '_resnet1D'
            pass
        else:
            gen = Generator(*args_gen)
            disc = Discriminator(*args_disc)
        return gen.float().to(device), disc.float().to(device)

class GAN(MODEL):
    """To init non-saturating GAN class (Indeed, the baseline will not be the minmax GAN)"""

    def __init__(self, parser, path):
        super().__init__(parser, path)
        self.BCE_loss = nn.BCELoss()
        
    def train(self):
        super().train()
        self.generator.weights_init()
        self.discriminator.weights_init()
        count = 0
        for epoch in range(self.epoch):
            with tqdm(self.train_loader, bar_format="{l_bar}{bar}{n_fmt}/{total_fmt}, ETA:{remaining}{postfix}", ncols=80, desc="Epoch " + str(epoch)) as t:
                mean_loss_D, mean_loss_G, n = 0, 0, 0
                count += 1
                for img in t:
                    n += 1
                    # Discriminator
                    self.discriminator.zero_grad()
                    img = img.float().to(self.device)
                    y = define_labels_GAN(img.size(0), True).to(self.device)
                    pred = self.discriminator(img)
                    lossD_real = self.BCE_loss(pred, y)
                    lossD_real.backward()
                    D_x = pred.mean().item()
                    latent_zs = torch.randn(img.size(0), self.latent_size, device=self.device)
                    gen = self.generator(latent_zs)
                    y_fake = define_labels_GAN(img.size(0), False).to(self.device)
                    pred_fake = self.discriminator(gen.detach())
                    lossD_fake = self.BCE_loss(pred_fake, y_fake)
                    lossD_fake.backward()
                    D_g_z = pred.mean().item()

                    lossD = lossD_real + lossD_fake
                    self.doptimizer.step()

                    # Generator
                    self.generator.zero_grad()
                    y.fill_(1)
                    pred = self.discriminator(gen).view(-1)
                    lossG = self.BCE_loss(pred, y)
                    lossG.backward()
                    D_g_z_update = pred.mean().item()
                    self.goptimizer.step()

                    t.set_postfix({"loss_D": "{0:.3f}".format(lossD.item()), "loss_G": "{0:.3f}".format(lossG.item())})
                    self.hist_train['loss_d_batch'].append(lossD.item())
                    self.hist_train['loss_g_batch'].append(lossG.item())
                    mean_loss_D = ((n-1) * mean_loss_D + lossD.tolist()) / n
                    mean_loss_G = ((n-1) * mean_loss_G + lossG.tolist()) / n

                if count % 1000 == 0:
                    self.hist_train['loss_d_epoch'].append(mean_loss_D)
                    self.hist_train['loss_g_epoch'].append(mean_loss_G)
                    with torch.no_grad():
                       gen = self.generator(self.fixed_noise).detach().cpu()
                       self.hist_train['fixed_noises'].append(gen)
        print("Training finished :)")


class GAN_GP(MODEL):
    def __init__(self, parser, path):
        """ To init a NS GAN with gradient penalty - http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf"""
        super().__init__(parser, path)
        self.lambda_gp = parser.get('lambda_gp')
        self.model = gan_gp(self.generator, self.discriminator, self.train_loader, optimizer_D=self.doptimizer,
                      optimizer_G=self.goptimizer, nz=self.latent_size, device=self.device,
                      ngpu=1, fixed_noise_size=64, nr_epochs=self.epoch, save_every=1000,
                      print_every=1, init_weights=True)

    def train(self):
        """Training function for NS GAN with gradient penalty objects -> No parameters"""
        super().train()
        self.model.train(disc_iters=self.d_iters, long_disc_iters=self.long_d_iters, lambda_gp=self.lambda_gp)
        img_list, D_losses, G_losses = self.model.get_training_results()
        self.hist_train['loss_d'] = D_losses
        self.hist_train['loss_g'] = G_losses
        self.hist_train['fixed_noises'] = img_list
        print("Training finished :)")


class WGAN(MODEL):
    def __init__(self, parser, path):
        """ To init a wasserstein GAN - https://arxiv.org/pdf/1701.07875.pdf"""
        super().__init__(parser, path)
        self.goptimizer = optim.RMSprop(self.generator.parameters(), lr=self.lr_g)
        self.doptimizer = optim.RMSprop(self.discriminator.parameters(), lr=self.lr_d)
        self.model = wgan(self.generator, self.discriminator, self.train_loader, optimizer_D=self.doptimizer,
                      optimizer_G=self.goptimizer, nz=self.latent_size, device=self.device,
                      ngpu=1, fixed_noise_size=64, nr_epochs=self.epoch, save_every=1000,
                      print_every=1, init_weights=True)

    def train(self):
        """Training function for wasserstein GAN objects -> No parameters"""
        super().train()
        self.model.train(critic_iters=self.d_iters, long_critic_iters=self.long_d_iters)
        img_list, D_losses, G_losses = self.model.get_training_results()
        self.hist_train['loss_d'] = D_losses
        self.hist_train['loss_g'] = G_losses
        self.hist_train['fixed_noises'] = img_list
        print("Training finished :)")


class WGAN_GP(MODEL):
    def __init__(self, parser, path):
        """ To init a wasserstein GAN with gradient penalty - http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf"""
        super().__init__(parser, path)
        self.lambda_gp = parser.get('lambda_gp')
        self.model = wgan_gp(self.generator, self.discriminator, self.train_loader, optimizer_D=self.doptimizer,
                      optimizer_G=self.goptimizer, nz=self.latent_size, device=self.device,
                      ngpu=1, fixed_noise_size=64, nr_epochs=self.epoch, save_every=1000,
                      print_every=1, init_weights=True)

    def train(self):
        """Training function for wasserstein GAN with gradient penalty objects -> No parameters"""
        super().train()
        self.model.train(critic_iters=self.d_iters, long_critic_iters=self.long_d_iters, lambda_gp=self.lambda_gp)
        img_list, D_losses, G_losses = self.model.get_training_results()
        self.hist_train['loss_d'] = D_losses
        self.hist_train['loss_g'] = G_losses
        self.hist_train['fixed_noises'] = img_list
        print("Training finished :)")

class WGAN_GP_CT(MODEL):
    def __init__(self, parser, path):
        """ To init a wasserstein GAN with gradient penalty and a consistency term - https://arxiv.org/pdf/1803.01541.pdf"""
        super().__init__(parser, path)
        self.lambda_gp = parser.get('lambda_gp')
        self.lambda_gp_ct = parser.get('lambda_gp_ct')
        self.m_param = parser.get('m_param')
        self.model = wgan_gp_ct(self.generator, self.discriminator, self.train_loader, optimizer_D=self.doptimizer,
                      optimizer_G=self.goptimizer, nz=self.latent_size, device=self.device,
                      ngpu=1, fixed_noise_size=64, nr_epochs=self.epoch, save_every=1000,
                      print_every=1, init_weights=True)

    def train(self):
        """Training function for wasserstein GAN with gradient penalty and a consistency term objects -> No parameters"""
        super().train()
        self.model.train(critic_iters=self.d_iters, long_critic_iters=self.long_d_iters, lambda_1=self.lambda_gp, lambda_2=self.lambda_gp_ct,
                         M=self.m_param)
        img_list, D_losses, G_losses = self.model.get_training_results()
        self.hist_train['loss_d'] = D_losses
        self.hist_train['loss_g'] = G_losses
        self.hist_train['fixed_noises'] = img_list
        print("Training finished :)")
