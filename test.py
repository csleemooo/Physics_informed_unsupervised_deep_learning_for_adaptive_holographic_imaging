from model.Forward_operator import Holo_Generator
from model.Inverse_operator import Distance_Generator, Field_Generator, Discriminator

import matplotlib.pyplot as plt
import torch
from torch import nn
from math import pi
from functions.utils import center_crop, make_path
import os
import numpy as np
from model.Initialization import parse_args
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from functions.metrics import fsim
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    args = parse_args()
    args.batch_size=10

    if args.num_depth == 5:
        train_holo_list = list(range(10, 20, 2))
    else:
        train_holo_list = [14]

    test_holo_list = list(i/2 for i in range(20, 40, 1))
    args.distance_min, args.distance_max = 10, 18

    args.phase_normalize = 2 * pi
    args.distance_normalize=args.distance_max - args.distance_min
    args.distance_normalize_constant=args.distance_min/args.distance_normalize

    args.save_folder = args.data_name + '_' + args.mode + '_' + str(len(train_holo_list)) + 'depth'
    test_result_path = args.test_result_root

    data_path = os.path.join(args.data_root, args.data_name)
    saving_path = os.path.join(args.result_root, args.save_folder)
    model_saving_path = os.path.join(args.model_root, args.save_folder)

    make_path(test_result_path)
    test_result_path = os.path.join(test_result_path, args.save_folder)
    make_path(test_result_path)

    ##############################################################
    transform_img = transforms.Compose([transforms.Resize([112, 112]), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
         transforms.Grayscale(), transforms.ToTensor()])
    test_gt_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=False, transform=transform_img)
    test_holo_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=False, transform=transform_img)

    args.phase_normalize = pi
    ##############################################################


    # define model
    holo_G = Holo_Generator(args).to(device)
    distance_G = Distance_Generator(args).to(device)
    Field_G = Field_Generator(args).to(device)

    params=torch.load(os.path.join(model_saving_path, "model.pth"))

    Field_G.load_state_dict(params['Field_G_state_dict'])
    distance_G.load_state_dict(params['distance_G_state_dict'])


    Field_G.eval()
    distance_G.eval()

    test_gt = DataLoader(test_gt_loader, batch_size=args.batch_size, shuffle=True)
    test_holo = DataLoader(test_holo_loader, batch_size=args.batch_size, shuffle=True)

    N_test = test_holo_loader.__len__()//args.batch_size

    # plot training loss
    fig = plt.figure(1, figsize=[12, 8])
    for idx, (loss_name, loss_val) in enumerate(params['loss'].items()):
        plt.subplot(2, 2, idx+1)
        plt.plot(loss_val)
        plt.xlabel('iterations')
        plt.ylabel(loss_name)

    fig.savefig(os.path.join(test_result_path, 'Loss.png'))
    plt.close(fig)
    print('Loss plot is saved at: %s'%(os.path.join(test_result_path, 'Loss.png')))

    make_path(os.path.join(test_result_path, 'test_image'))

    real_distance_list, predict_distance_list = [], []
    fsim_amplitude, pcc_amplitude = [], []
    fsim_phase, pcc_phase = [], []

    for b, ([gt_batch,_], [holo_batch,_]) in enumerate(zip(test_gt, test_holo)):
        if args.data_name == 'mnist':

            if args.mode == 'phase':
                real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                real_amplitude = torch.ones_like(real_phase).to(device) * 0.6

            elif args.mode == 'amplitude':
                real_amplitude = (1 - F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6
                real_phase = torch.zeros_like(real_amplitude).to(device)

            elif args.mode == 'complex_amplitude':
                real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                gt_batch, _ = next(iter(test_gt))
                real_amplitude = (1 - F.pad(holo_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6

            # d = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device).float()

            d = torch.randint(low=args.distance_min, high=args.distance_max,
                              size=(args.batch_size, 1, 1, 1)).to(device).float()
            d = -args.distance_normalize_constant + d / args.distance_normalize

            holo = holo_G(real_amplitude, real_phase, d).detach().to(device).float()

            real_phase *= args.phase_normalize

        ## generate test amplitude and distance
        fake_amplitude, fake_phase = Field_G(holo)
        fake_distance = distance_G(holo)
        fake_holo = holo_G(fake_amplitude, fake_phase, fake_distance)

        fake_distance = (fake_distance + args.distance_normalize_constant)* args.distance_normalize
        real_distance = (d + args.distance_normalize_constant)* args.distance_normalize

        for ra, fa, rp, fp, rd, fd in zip(real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance):

            ra = ra.squeeze().cpu().detach().numpy()
            fa = fa.squeeze().cpu().detach().numpy()
            rp = rp.squeeze().cpu().detach().numpy()
            fp = fp.squeeze().cpu().detach().numpy()

            fsim_amplitude.append(fsim(ra, fa))
            fsim_phase.append(fsim(rp, fa))
            pcc_amplitude.append(pearsonr(np.ravel(ra), np.ravel(fp)))
            pcc_phase.append(pearsonr(np.ravel(rp), np.ravel(fp)))

            real_distance_list.append(rd.item())
            predict_distance_list.append(fd.item())

        if b%10 == 0:
            fake_holo = fake_holo.cpu().detach().numpy()[0][0]
            holo = holo.cpu().detach().numpy()[0][0]

            real_amplitude = real_amplitude.cpu().detach().numpy()[0][0]
            real_phase = real_phase.cpu().detach().numpy()[0][0]
            fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0] / args.amplitude_normalize
            fake_phase = fake_phase.cpu().detach().numpy()[0][0] * args.phase_normalize

            fig2 = plt.figure(2, figsize=[12, 8])

            plt.subplot(2, 3, 1)
            plt.title('input holography')
            plt.imshow(holo, cmap='gray', vmax=1.0, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)
            plt.subplot(2, 3, 2)
            plt.title('ground truth\nmeasured:%2.2fmm'%real_distance[0].item())
            plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)
            plt.subplot(2, 3, 3)
            plt.title('reconstructed amplitude\npredict:%2.2fmm'%fake_distance[0].item())
            plt.imshow(fake_amplitude, cmap='gray', vmax=1, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)

            plt.subplot(2, 3, 4)
            plt.title('generated_holography')
            plt.imshow(fake_holo, cmap='gray', vmax=1.0, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)
            plt.subplot(2, 3, 5)
            plt.title('ground truth phase')
            plt.imshow(real_phase, cmap='jet', vmax=pi, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)
            plt.subplot(2, 3, 6)
            plt.title('reconstructed phase')
            plt.imshow(fake_phase, cmap='jet', vmax=pi, vmin=0)
            plt.axis('off')
            plt.colorbar(shrink=0.75)

            fig_save_name = os.path.join(os.path.join(test_result_path, 'test_image'), 'test%d.png'%(b+1))
            fig2.savefig(fig_save_name)
            print('Test %d is saved at: %s' % (b+1, fig_save_name))
            plt.close(fig2)

        if (b+1)%100 == 0:
            y = [np.mean(fsim_amplitude), np.mean(pcc_amplitude), np.mean(fsim_phase), np.mean(pcc_phase)]
            std = [np.std(fsim_amplitude), np.std(pcc_amplitude), np.std(fsim_phase), np.std(pcc_phase)]
            x = list(range(y))

            fig = plt.figure(1, figsize=[8, 8])
            plt.title("Image metric")
            plt.errorbar(x, y, yerr=std)
            plt.xticks(x, ['FSIM\namplitude', 'PCC\namplitude', 'FSIM\nphase', 'PCC\nphase'])
            plt.ylabel('prediction')

            fig_save_name = os.path.join(os.path.join(test_result_path), 'Image_metric.png')
            fig.savefig(fig_save_name)
            print('Image metric test is saved at: %s' % fig_save_name)
            plt.close(fig)

            break

    mae = np.mean(np.abs(np.array(real_distance_list) - np.array(predict_distance_list)))
    r2 = r2_score(real_distance_list, predict_distance_list)

    d_ = {i:[] for i in np.unique(real_distance_list)}
    for rd, fd in zip(real_distance_list, predict_distance_list):
        d_[rd].append(fd)

    x = [i for i in d_.keys()]
    y = [np.mean(i) for i in d_.values()]
    std = [np.std(i) for i in d_.values()]

    fig=plt.figure(1, figsize=[8,8])
    plt.title("Distance prediction errorbar\nMAE:%1.2f R_square:%1.3f"%(mae, r2))
    plt.errorbar(x, y, yerr=std)
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    fig_save_name = os.path.join(os.path.join(test_result_path), 'distance_test_errorbar.png')
    fig.savefig(fig_save_name)
    print('Distance test is saved at: %s' % fig_save_name)
    plt.close(fig)