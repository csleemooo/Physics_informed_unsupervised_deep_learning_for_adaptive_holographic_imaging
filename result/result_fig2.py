import os, sys
sys.path.append(os.getcwd())

from model.Inverse_operator import Distance_Generator, Field_Generator

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from functions.Data_Loader_custom import Holo_Recon_Dataloader_supervised
from functions.utils import make_path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

if __name__ == '__main__':
    device = torch.device('cpu')

    model_path = "./model_parameters/polystyrene_bead_6depth"
    data_path = './data/fig2_polystyrene_bead'
    test_result_path = './test_result'

    make_path(test_result_path)
    test_result_path = os.path.join(test_result_path, 'fig2_polystyrene_bead')
    make_path(test_result_path)

    # define model
    model = torch.load(os.path.join(model_path, 'proposed', 'model.pth'))
    args = model['args']

    Field_G = Field_Generator(args)
    Field_G.load_state_dict(model['Field_G_state_dict'])
    Field_G.eval()

    distance_G = Distance_Generator(args)
    distance_G.load_state_dict(model['distance_G_state_dict'])
    distance_G.eval()

    transform_img = transforms.Compose([transforms.ToTensor()])
    holo_input, amplitude_result, phase_result, distance_result, real_distance = [], [], [], [], []
    for t in [7, 17]:
        print('Complex amplitude reconstruction from hologram intensity measured at %dmm.' % t)
        trained_model = []
        test_loader = Holo_Recon_Dataloader_supervised(root=data_path, data_type=['holography'], image_set='test',
                                                 transform=transform_img, holo_list=[t])

        holo, real_amplitude, real_phase = next(iter(DataLoader(test_loader, batch_size=1, shuffle=False)))

        # to match the global phase, subtract mean value
        real_phase = real_phase.detach().numpy()[0][0]
        real_phase -= np.mean(real_phase)

        fake_amplitude, fake_phase = Field_G(holo.float())

        fake_distance = distance_G(holo.float())

        distance_result.append((fake_distance.item() + args.distance_normalize_constant) * args.distance_normalize)
        real_distance.append(t)
        holo_input.append(holo.detach().numpy()[0][0])
        amplitude_result.append(fake_amplitude.detach().numpy()[0][0] / args.amplitude_normalize)

        fake_phase = fake_phase.detach().numpy()[0][0] * args.phase_normalize

        fake_phase -= np.mean(fake_phase) # to match the global phase, subtract mean value
        phase_result.append(fake_phase)

    else:

        fig = plt.figure(1, figsize=[10, 10])

        for f_idx, (holo, amplitude, phase, distance_pred, distance_real) in \
                enumerate(zip(holo_input, amplitude_result, phase_result, distance_result, real_distance)):
            plt.subplot(3, 3, 1+f_idx)
            plt.title('hologram intensity \n measured:%dmm' % distance_real)
            plt.imshow(holo, cmap='gray', vmax=0.4, vmin=0)
            plt.axis('off')
            plt.subplot(3, 3, 4+f_idx)
            plt.title('Reconstructed amplitude\n predict:%2.2fmm'%distance_pred)
            plt.imshow(amplitude, cmap='gray', vmax=0.7, vmin=0)
            plt.axis('off')
            plt.subplot(3, 3, 7+f_idx)
            plt.title('Reconstructed phase')
            plt.imshow(phase, cmap='hot', vmax=2.5, vmin=-0.1)
            plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.title('Ground truth amplitude')
        plt.imshow(real_amplitude.detach().numpy()[0][0], cmap='gray', vmax=0.7, vmin=0)
        plt.axis('off')
        plt.subplot(3, 3, 9)
        plt.title('Ground truth phase')
        plt.imshow(real_phase, cmap='hot', vmax=2.5, vmin=-0.1)
        plt.axis('off')

        fig.savefig(os.path.join(test_result_path, 'result_image.png'))
        print('Result figure is saved at %s.' % os.path.join(test_result_path, 'result_image.png'))

        result_data = {'input_hologram_intensity': holo_input,
                       'Reconstructed_amplitude': amplitude_result,
                       'Reconstructed_phase': phase_result,
                       'Reconstructed_distance': distance_result,
                       'Ground_truth_distance': real_distance,
                       'Ground_truth_amplitude': real_amplitude.detach().numpy()[0][0],
                       'Ground_truth_phase': real_phase}

        sio.savemat(os.path.join(test_result_path, 'result.mat'), result_data)
        print('Result data is saved at %s.' % os.path.join(test_result_path, 'result.mat'))

        plt.show()