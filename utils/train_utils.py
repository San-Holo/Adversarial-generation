import imageio
import json
from functools import partial
from vegans_modified.gans import GAN, GAN_GP, GAN_GP_CT, WGAN, WGAN_GP, WGAN_GP_CT



def load_parsers(campaign_name):
    """Load a experiment campaign stored as a JSON file

    Parameters
    ----------
    camaing_name : string
        File path to the JSON file
        
    Returns
    -------
    Tuple of dicts, one for each gan framework described in the JSON file
    """
    parsers = json.loads(open(campaign_name).read()).values()
    return tuple(parsers)
    

def pipeline_gan(parser, path, train=False):
    """GAN "Pipeline" from data loading to results. The two lists possibly used when creating
        the GAN object are based on the original DCGAN paper. For any questions about their
        modification, please refer to build_network_utils.py

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters

        path : String
            Where the trained generator and discriminator will be saved, and results too

        train : Boolean
            If False, it will just create the object and visualize its content as a sanity check
        """
    gan = GAN(parser, path)
    if train:
        gan.train()
        gan.save(parser)

def pipeline_gan_gp(parser, path, train=False):
    """NS GAN with gradient penalty  "Pipeline" from data loading to results. The two lists possibly used when creating
        the GAN object are based on the original DCGAN paper. For any questions about their
        modification, please refer to build_network_utils.py

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters

        path : String
            Where the trained generator and discriminator will be saved, and results too

        train : Boolean
            If False, it will just create the object and visualize its content as a sanity check
        """
    gan = GAN_GP(parser, path)
    if train:
        gan.train()
        gan.save(parser)


def pipeline_wgan(parser, path, train=False):
    """WGAN "Pipeline" from data loading to results. The two lists possibly used when creating
        the WGAN object are based on the original DCGAN paper. For any questions about their
        modification, please refer to build_network_utils.py.

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters

        path : String
            Where the trained generator and discriminator will be saved, and results too

        train : Boolean
            If False, it will just create the object and visualize its content as a sanity check
        """
    wgan = WGAN(parser, path)
    if train:
        wgan.train()
        wgan.save(parser)

def pipeline_wgan_gp(parser,  path, train=False):
    """WGAN_GP "Pipeline" from data loading to results. The two lists possibly used when creating
        the WGAN_GP object are based on the original DCGAN paper. For any questions about their
        modification, please refer to build_network_utils.py

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters

        path : String
            Where the trained generator and discriminator will be saved, and results too
            
        train : Boolean
            If False, it will just create the object ant visualize its content as a sanity check

        """
    wgan_gp = WGAN_GP(parser, path)
    if train:
        wgan_gp.train()
        wgan_gp.save(parser)

def pipeline_wgan_gp_ct(parser,  path, train=False):
    """WGAN_GP_CT "Pipeline" from data loading to results. The two lists possibly used when creating
        the WGAN_GP object are based on the original DCGAN paper. For any questions about their
        modification, please refer to build_network_utils.py

        Parameters
        ----------
        parser : dict
            Contains all hyperparameters

        path : String
            Where the trained generator and discriminator will be saved, and results too
        
        train : Boolean
            If False, it will just create the object ant visualize its content as a sanity check
        """
    wgan_gp_ct = WGAN_GP_CT(parser, path)
    if train:
        wgan_gp_ct.train()
        wgan_gp_ct.save(parser)

# Not nice, must be refactored #

funcs = {'gan': partial(pipeline_gan), 'gan_gp': partial(pipeline_gan_gp), 'gan_gp_ct': partial(pipeline_gan_gp_ct),
         'wgan': partial(pipeline_wgan), 'wgan_gp': partial(pipeline_wgan_gp), 'wgan_gp_ct': partial(pipeline_wgan_gp_ct)}

# Main pipeline #

def pipeline(method, parser, path, train=False):
    """Final pipeline method -> Must be used all the time

    Parameters
    ----------
    method : string
        Which gan framework will be used -> 'gan', 'gan_gp', gan_gp_ct', 'wgan', 'wgan_gp', 'wgan_gp_ct'

    parser : dict
        Contains all hyperparameters

    path : String
        Where the trained generator and discriminator will be saved, and results too
    """
    pipe = funcs.get(method)
    pipe(parser, path, train)
