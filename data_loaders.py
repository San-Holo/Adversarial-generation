import sys

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shutil
import os
from glob import iglob, glob
from skimage import io
import pickle

from torch.utils.data.dataset import Dataset

np.random.seed(50)
torch.manual_seed(50)

# Global variables #
####################


DIR_DATA = '.'
DIR_IMA = '.'
        


# Utils #
#########

def load_pickle(file):
    """Util to load pkl file

    Parameters
    ----------
    file : String
        A pickle file
        
    Returns
    --------
        Content as a pickle object
    """
    with open(file, "rb") as f_in:
            content = pickle.load(f_in)
    return content

# Loading #
###########


def load_images(dir_path, ext='png', size=None):
    """Load images from dirpath -> without normalization

    Parameters
    ----------
    dir_path : String
        Directory where all images are gathered

    ext : String
        Required file type. If you want to load images with strictly more than 4
        channels, use 'pkl'

    size : None or int
        Percentage of the returned dataset. If None, :: is retrieved, if int, ::size is
        
    Returns
    --------
        Tensor with all images, size : (nb_images, channels, height, width)
    """
    if ext == 'png':
        images = np.array(list(map(io.imread, iglob(dir_path + '\\' + '*png'))))
    else:
        images = np.array(list(map(load_pickle, iglob(dir_path + '\\' + '*pkl'))))
    images = images[...]
    print(images.shape)
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 2, 3) #Back to good order : channel - height - width
    images = torch.tensor(images).float()
    #images = (images - images.mean()) / images.std() # Normalization such as this one need to be removed because we might loose some info. We will prefer the one above
    min_images = torch.min(images, dim=3, keepdim=True)[0].min(2, keepdim=True)[0].min(0, keepdim=True)[0]
    max_images = torch.max(images, dim=3, keepdim=True)[0].max(2, keepdim=True)[0].max(0, keepdim=True)[0]
    span = max_images - min_images
    images = (2. * (images - min_images) / span) - 1.
    if size == None:
        return images.float(), images.size()[1:]
    else:
        return images[::size].float(), images[::size]

def define_labels_GAN(dim, real):
    """Define labels for the GANs, not used in Dataset object. It will be used only during training process

    Parameters
    ----------
    dim : int
        Used to create the tensor

    real : boolean
        If True, we'll fill the wanted tensor with ones, If not, with zeros (real or fake data)
            
    Returns
    --------
        Tensor of shape (dim, 1)
    """
    if real:
        return torch.ones(dim)
    else:
        return torch.zeros(dim)

# Data types #
##############

class SeriesAsImagesDataset(Dataset):
    """A class to create our own images dataset which can be used with DataLoader from torch"""
        
    def __init__(self, dir_path = DIR_IMA, method = 'grey', ext='png', size=None):
        """Loading and assigning data in order to create a dataset of images

        Parameters
        ----------
        dir_path :  String
            A path to the directory in which each image is stored

        method : String
            3 possible values : 'grey', 'rgb_grey', 'cmu_y' -> Which image representation we'll use

        ext : String
            2 possible values 'png', 'pkl' -> pkl is used when the number of channels > 4

        size : float
            Which percentage of data is used for both training and evaluation
        """
        if method == 'grey':
            path = os.path.join(dir_path,'Greyscale')
        elif method == 'rgb_grey':
            path = os.path.join(dir_path,'RGB_greyscale')
        elif method == 'cmu_y':
            path = os.path.join(dir_path,'CMYK_greyscale')
        else:
            raise NameError
        self.images, self.images_size = load_images(path, ext, size)

    def __getitem__(self, index):
        """Loading and assigning data in order to create dataset. Don't need labels because of unsupervised GAN framework

        Parameters
        ----------
        index :  int
            Nth data in the dataset

        Returns
        ---------
        x : tensor
            Entry for learning -> img
        """

        return self.images[index]


    def __len__(self):
        """Returning number of datas"""
        return len(self.images)



