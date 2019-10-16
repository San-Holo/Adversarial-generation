# Adversarial-generation

Small exploration of generative models, starting with GANs and slowly going towards manifold based methods.
The structure of each GAN model is based on a github repo, from which I added some changes. I would like to thank its author and provide a link for it: https://github.com/unit8co/vegans.

This repo is a tool that I use for seeral purposes, including a vanilla GAN, a wasserstein GAN and their regularized versions using gradient penalty and a consistency term. Moreover, several normalizations methods are available including the spectral one, as well as common technics such as dropout.  

Everything is implemented with PyTorch, using its data structures to provide pipelines useful for training and inference. There is a "pipeline" for each type of model, and the former uses a JSON file to retrieve all wanted hyperparameters (see the example file in parsers). So, to train a new model, you just have to call one of the functions presented in the main python file, and change the written path to a parsed decription, if needed.

Upcoming models:

LSGAN
SAGAN
BIGAN

