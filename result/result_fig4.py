import os, sys
sys.path.append(os.getcwd())

from model.Inverse_operator import Distance_Generator, Field_Generator

import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from functions.Data_Loader_custom import Holo_Recon_Dataloader
from functions.utils import make_path
from math import pi
import matplotlib.pyplot as plt

import scipy.io as sio

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if __name__ == '__main__':


    model_path = "./model_parameters/red_blood_cells"
    data_path = './data/fig4_red_blood_cell'
    test_result_path = './test_result'

    make_path(test_result_path)
    test_result_path = os.path.join(test_result_path, 'fig4_red_blood_cell')
    make_path(test_result_path)

    model = torch.load(os.path.join(model_path, 'model.pth'))
    args = model['args']

    distance_G = Distance_Generator(args)
    Field_G = Field_Generator(args)

    Field_G.load_state_dict(model['Field_G_state_dict'])
    distance_G.load_state_dict(model['distance_G_state_dict'])

    distance_G.eval()
    Field_G.eval()

    ## load test data
    transform_img = transforms.Compose([transforms.ToTensor()])

    test_holo_loader = Holo_Recon_Dataloader(root=data_path, data_type=['holography'], image_set='test',
                                             transform=transform_img, holo_list=[1], sort=True)
    N_test = test_holo_loader.__len__()
    ##############################################################


    test_holo = iter(DataLoader(test_holo_loader, batch_size=1, shuffle=False))
    holo_list, amplitude_list, phase_list = [], [], []
    frames=[29, 39, 58, 78, 85]
    for b in range(N_test):

        holo = next(test_holo).float()

        fake_amplitude, fake_phase = Field_G(holo)
        fake_distance = distance_G(holo)
        fake_distance = (fake_distance.item() + args.distance_normalize_constant) * args.distance_normalize

        fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0] / args.amplitude_normalize
        fake_phase = fake_phase.cpu().detach().numpy()[0][0] * args.phase_normalize
        fake_phase -= np.mean(fake_phase)

        holo_list.append(holo.detach().numpy()[0][0])
        amplitude_list.append(fake_amplitude)
        phase_list.append(fake_phase)

    else:
        fig = plt.figure(1, figsize=[12, 8])

        for idx, (holo, amplitude, phase) in enumerate(zip(holo_list, amplitude_list, phase_list)):

            plt.subplot(3, 5, idx+1)
            plt.title('Frame %d'%frames[idx])
            plt.imshow(holo, cmap='gray', vmax=0.4, vmin=0)
            plt.axis('off')
            plt.subplot(3, 5, idx + 6)
            plt.imshow(amplitude, cmap='gray', vmax=0.7, vmin=0)
            plt.axis('off')
            plt.subplot(3, 5, idx + 11)
            plt.imshow(phase, cmap='hot', vmax=pi, vmin=0)
            plt.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(test_result_path, 'result_image.png'))
        print('Result figure is saved at %s.' % os.path.join(test_result_path, 'result_image.png'))

        result_data = {'input_hologram_intensity': holo_list,
                       'Reconstructed_amplitude': amplitude_list,
                       'Reconstructed_phase': phase_list}

        sio.savemat(os.path.join(test_result_path, 'result.mat'), result_data)
        print('Result data is saved at %s.' % os.path.join(test_result_path, 'result.mat'))

        plt.show()