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

    model_path = "./model_parameters/polystyrene_bead_1depth"
    data_path = './data/fig3_polystyrene_bead'
    test_result_path = './test_result'

    make_path(test_result_path)
    test_result_path = os.path.join(test_result_path, 'fig3_polystyrene_bead')
    make_path(test_result_path)

    transform_img = transforms.Compose([transforms.ToTensor()])
    for t in [8, 16]:

        trained_model = ['unet', 'cyclegan', 'phasegan', 'proposed']
        test_loader = Holo_Recon_Dataloader_supervised(root=data_path, data_type=['holography'], image_set='test',
                                                 transform=transform_img, holo_list=[t])

        holo, real_amplitude, real_phase = next(iter(DataLoader(test_loader, batch_size=1, shuffle=False)))

        # to match the global phase, subtract mean value
        real_phase = real_phase.detach().numpy()[0][0]
        real_phase -= np.mean(real_phase)

        print('Complex amplitude reconstruction from hologram intensity measured at %dmm.'%t)
        amplitude_result, phase_result, distance_result = [], [], []

        for m in trained_model:

            model = torch.load(os.path.join(model_path, m, 'model.pth'))
            args = model['args']

            Field_G = Field_Generator(args)
            Field_G.load_state_dict(model['Field_G_state_dict'])
            Field_G.eval()

            if m == 'proposed':
                distance_G = Distance_Generator(args)
                distance_G.load_state_dict(model['distance_G_state_dict'])
                distance_G.eval()

            fake_amplitude, fake_phase = Field_G(holo.float())

            if m == 'proposed':
                fake_distance = distance_G(holo.float())
                distance_result.append((fake_distance.item() + args.distance_normalize_constant) * args.distance_normalize)
            else:
                distance_result.append(0)

            amplitude_result.append(fake_amplitude.detach().numpy()[0][0] / args.amplitude_normalize)

            fake_phase = fake_phase.detach().numpy()[0][0] * args.phase_normalize
            # to match the global phase, subtract mean value
            fake_phase -= np.mean(fake_phase)
            phase_result.append(fake_phase)

        else:
            amplitude_result.append(real_amplitude.detach().numpy()[0][0])
            phase_result.append(real_phase)
            distance_result.append(0)

            fig = plt.figure(t, figsize=[18, 6])
            plt.subplot(2, 6, 1)
            plt.title('hologram intensity\n measured:%dmm'%t)
            plt.imshow(holo.detach().numpy()[0][0], cmap='gray', vmax=0.4, vmin=0)
            plt.axis('off')

            for f_idx, (model_name, amplitude, phase, distance) in enumerate(zip(trained_model + ['ground truth'], amplitude_result, phase_result, distance_result)):

                plt.subplot(2, 6, f_idx+2)
                plt.title(model_name + ' amplitude\n predict:%2.2fmm'%distance if distance else model_name + ' amplitude')
                plt.imshow(amplitude, cmap='gray', vmax=0.7, vmin=0)
                plt.axis('off')
                plt.subplot(2, 6, f_idx+2+6)
                plt.title(model_name + ' phase')
                plt.imshow(phase, cmap='hot', vmax=2.5, vmin=-0.1)
                plt.axis('off')

            fig.tight_layout()
            fig.savefig(os.path.join(test_result_path, '%dmm.png'%t))

            print('Result figure is saved at %s.' % os.path.join(test_result_path, '%dmm.png'%t))

            result_data = {'model_name': trained_model + ['ground truth'],
                           'input_hologram_intensity': holo.detach().numpy()[0][0],
                           'Reconstructed_amplitude': amplitude_result,
                           'Reconstructed_phase': phase_result,
                           'Reconstructed_distance':distance_result,
                           'Ground_truth_distance': t}

            sio.savemat(os.path.join(test_result_path, '%dmm.mat'%t), result_data)
            print('Result data is saved at %s.' % os.path.join(test_result_path, '%dmm.mat'%t))

    plt.show()

