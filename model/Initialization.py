import torch
import argparse
from math import pi
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='mnist', type=str) # polystyrene_bead, tissue_array, red_blood_cell, mnist
    parser.add_argument("--num_depth", default=5, type=int)  # 1 or 5 or 6 (5 for mnist 6 for polystyrene bead)
    parser.add_argument("--data_root", default='.\\data', type=str, help="Path to data folder")
    parser.add_argument("--result_root", default='.\\training_result', type=str, help="Path to save folder")
    parser.add_argument("--mode", default='complex_amplitude', type=str)

    # network type
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--batch_mode", default='G', type=str)
    parser.add_argument("--patchGAN", default=False, type=bool)
    parser.add_argument("--fc_layer", default=False, type=bool)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument("--fc_layer_input_feature", default=2304, type=int)
    parser.add_argument("--initial_channel", default=64, type=int)
    parser.add_argument("--Holo_G_input", default='amp_pha', type=str)

    # hyper-parameter
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--iterations", default=20000, type=int)
    parser.add_argument("--chk_iter", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)  # 1e-4 is default
    parser.add_argument("--lr_decay_epoch", default=5, type=int)
    parser.add_argument("--lr_decay_rate", default=0.95, type=float)
    parser.add_argument("--cycle_regularizer", default=100, type=int)
    parser.add_argument("--distance_regularizer", default=100, type=int)
    parser.add_argument("--penalty_regularizer", default=20, type=int)
    parser.add_argument("--ssim_regularizer", default=10, type=int)
    parser.add_argument("--target_real", default=1.0, type=float)
    parser.add_argument("--target_fake", default=0.0, type=float)
    parser.add_argument("--amplitude_normalize", default=1.0, type=float)
    parser.add_argument("--phase_normalize", default=2*pi, type=float)

    # experiment parameter
    parser.add_argument("--wavelength", default=532e-9, type=float)
    parser.add_argument("--pixel_size", default=6.5e-6, type=float)
    parser.add_argument("--distance_min", default=14, type=int)
    parser.add_argument("--distance_max", default=31, type=int)
    parser.add_argument("--distance_normalize", default=16, type=int)
    parser.add_argument("--distance_normalize_constant", default=14/16, type=float)

    return parser.parse_args()


def weights_initialize_normal(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:

        # apply a normal distribution to the weights and a bias=0
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)


def weights_initialize_xavier_normal(m):

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
