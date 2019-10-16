import argparse
import os
import json

from utils.train_utils import *

#Don't forget to change it to where you want to save your trained models and results
SAVE_DIR = './Results'
    
if __name__== "__main__":
	
    # 2D - RGB #
    ############

    # Fully conv 
    # ----------

    
    # Example EXP - R (except LN in Wasserstein frameworks) - WITHOUT TTUR 
    parser_gan, parser_gan_gp, parser_gan_gp_ct, parser_wgan, parser_wgan_gp, parser_wgan_gp_ct = load_parsers('./parsers/parsers_2D/parsers_R_FC_2D')
    pipeline('gan', parser_gan, SAVE_DIR, train=True)
    pipeline('wgan', parser_wgan, SAVE_DIR, train=True)
    pipeline('wgan_gp', parser_wgan_gp, SAVE_DIR, train=True)
    pipeline('wgan_gp_ct', parser_wgan_gp_ct, SAVE_DIR, train=True)
    pipeline('gan_gp', parser_gan_gp, SAVE_DIR, train=True)
    pipeline('gan_gp_ct', parser_gan_gp_ct, SAVE_DIR, train=True)
