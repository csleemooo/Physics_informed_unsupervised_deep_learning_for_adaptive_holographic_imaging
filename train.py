from model.Forward_operator import Holo_Generator
from model.Inverse_operator import Distance_Generator, Field_Generator, Discriminator

import matplotlib.pyplot as plt
import torch
from torch import nn
from math import pi
from functions.utils import center_crop, make_path
import os
from itertools import chain
import numpy as np
from model.Initialization import parse_args
from functions.gradient_penalty import calc_gradient_penalty
from functions.SSIM import SSIM
from functions.Data_Loader_custom import Holo_Recon_Dataloader
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    args = parse_args()

    if args.data_name == 'polystyrene_bead':
        args.distance_min, args.distance_max=6, 18
        args.phase_normalize = 2 * pi
        if args.num_depth == 6:
            train_holo_list = list(range(7,18,2))
        else:
            train_holo_list = [13]
        test_holo_list = list(range(7, 18, 2))

    elif args.data_name == 'tissue_array':
        args.distance_min, args.distance_max = 10, 28
        args.phase_normalize = 2 * pi
        train_holo_list = [19]
        test_holo_list = list(range(12, 28, 2))

    elif args.data_name == 'red_blood_cell':
        args.distance_min, args.distance_max = 18, 33
        args.phase_normalize = 3 * pi
        args.iterations = 35000
        train_holo_list = [1]
        test_holo_list = list(range(1, 17, 1))

    elif args.data_name == 'mnist':
        if args.num_depth == 5:
            train_holo_list = list(range(10, 20, 2))
        else:
            train_holo_list = [14]

        test_holo_list = list(range(10, 19, 1))
        args.distance_min, args.distance_max = 10, 18
        args.phase_normalize = 2 * pi
        args.iterations = 20000


    args.distance_normalize=args.distance_max - args.distance_min
    args.distance_normalize_constant=args.distance_min/args.distance_normalize

    args.save_folder = args.data_name+'_'+args.mode+'_'+str(len(train_holo_list))+'depth'

    data_path = os.path.join(args.data_root, args.data_name)
    saving_path = os.path.join(args.result_root, args.save_folder)
    model_saving_path = os.path.join(args.model_root, args.save_folder)

    ##############################################################
    if args.data_name == 'mnist':
        transform_img = transforms.Compose([transforms.Resize([112, 112]), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
             transforms.Grayscale(), transforms.ToTensor()])
        train_gt_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=True, transform=transform_img)
        train_holo_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=True, transform=transform_img)

        transform_img = transforms.Compose([transforms.Resize([112, 112]), transforms.Grayscale(), transforms.ToTensor()])
        test_gt_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=False, transform=transform_img)
        test_holo_loader = torchvision.datasets.MNIST(root=args.data_root, download=True, train=False, transform=transform_img)

        args.phase_normalize = pi
        N_test = test_holo_loader.__len__()

    else:
        # data loader set
        transform_img = transforms.Compose([transforms.ToTensor()])

        # train data loader
        train_gt_loader = Holo_Recon_Dataloader(root=data_path, data_type=['gt_amplitude', 'gt_phase'],
                                                image_set='train', transform=transform_img, train_type="train")
        train_holo_loader = Holo_Recon_Dataloader(root=data_path, data_type=['holography'],
                                                  image_set='train', transform=transform_img, train_type="train",
                                                  holo_list=train_holo_list)

        # test data loader
        test_gt_loader = Holo_Recon_Dataloader(root=data_path, data_type=['gt_amplitude', 'gt_phase'],
                                               image_set='test', transform=transform_img)
        test_holo_loader = Holo_Recon_Dataloader(root=data_path, data_type=['holography'], image_set='test',
                                                 transform=transform_img, holo_list=test_holo_list)
        N_test = test_holo_loader.__len__()
        ##############################################################

    # define model
    holo_G = Holo_Generator(args).to(device=device)
    distance_G = Distance_Generator(args).to(device=device)
    Field_G = Field_Generator(args).to(device=device)
    Field_D = Discriminator(args, input_channel=3).to(device=device)

    # optimizer
    op_G = torch.optim.Adam(chain(Field_G.parameters(), distance_G.parameters()), lr=1e-3, betas=(0.5, 0.9))
    op_D = torch.optim.Adam(Field_D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(op_G, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(op_D, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    # loss
    criterion_cycle = nn.L1Loss()
    criterion_MSE = nn.MSELoss()
    criterion_wgan = torch.mean
    criterion_ssim = SSIM()

    # loss list
    loss_G_list = []
    loss_distance_G_list = []
    loss_field_D_list = []
    loss_field_D_penalty_list=[]

    loss_sum_Field_D = 0
    loss_sum_Field_D_penalty = 0
    loss_sum_G = 0
    loss_sum_D = 0

    train_gt_loader = DataLoader(train_gt_loader, batch_size=args.batch_size, shuffle=True)
    train_holo_loader = DataLoader(train_holo_loader, batch_size=args.batch_size, shuffle=True)

    for it in range(args.iterations):

        Field_G.train()
        Field_D.train()
        distance_G.train()

        if args.data_name == 'mnist':
            gt_batch, _ = next(iter(train_gt_loader))
            holo_batch, _ = next(iter(train_holo_loader))

            if args.mode == 'phase':
                real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)*args.phase_normalize
                real_amplitude = torch.ones_like(real_phase).to(device)*0.6

                holo_phase = F.pad(holo_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                holo_amplitude = torch.ones_like(holo_phase).to(device)*0.6

            elif args.mode == 'amplitude':
                real_amplitude = (1 - F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6
                real_phase = torch.zeros_like(real_amplitude).to(device)*args.phase_normalize

                holo_amplitude = (1 - F.pad(holo_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6
                holo_phase = torch.zeros_like(holo_amplitude).to(device)

            elif args.mode == 'complex_amplitude':
                real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)*args.phase_normalize
                gt_batch, _ = next(iter(train_gt_loader))
                real_amplitude = (1 - F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device))*0.6

                holo_phase = F.pad(holo_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                holo_batch, _ = next(iter(train_holo_loader))
                holo_amplitude = (1 - F.pad(holo_batch, (56, 56, 56, 56), 'constant', 0).to(device))*0.6


            d = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device).float()

            holo = holo_G(holo_amplitude, holo_phase, d).detach()
            del holo_phase, holo_amplitude, d, gt_batch, holo_batch

        else:
            real_amplitude, real_phase = next(iter(train_gt_loader))
            holo = next(iter(train_holo_loader))

        real_amplitude = center_crop(real_amplitude, args.crop_size).to(device).float()* args.amplitude_normalize
        real_phase = center_crop(real_phase, args.crop_size).to(device).float()/ (args.phase_normalize)
        holo = center_crop(holo, args.crop_size).to(device).float()

        real_distance = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()

        fake_amplitude, fake_phase = Field_G(holo)
        fake_distance = distance_G(holo)

        consistency_holo = holo_G(fake_amplitude, fake_phase, fake_distance)
        fake_holo = holo_G(real_amplitude, real_phase, real_distance).float()

        consistency_amplitude, consistency_phase = Field_G(fake_holo)
        consistency_distance = distance_G(fake_holo)

        real_field = torch.cat([real_amplitude, real_phase], dim=1)
        fake_field = torch.cat([fake_amplitude, fake_phase], dim=1)
        consistency_field = torch.cat([consistency_amplitude, consistency_phase], dim=1)

        ## train discriminator
        op_D.zero_grad()

        fake_D = Field_D(fake_field.detach())
        real_D = Field_D(real_field)

        criterion_gradient_amplitude = calc_gradient_penalty(Field_D, real_field, fake_field, real_amplitude.shape[0])
        loss_field_D = criterion_wgan(fake_D.mean(dim=(-2,-1))) - criterion_wgan(real_D.mean(dim=(-2,-1))) \
                           + args.penalty_regularizer*criterion_gradient_amplitude

        loss_sum_Field_D_penalty += args.penalty_regularizer*criterion_gradient_amplitude.item()
        loss_sum_Field_D += loss_field_D.item() - args.penalty_regularizer*criterion_gradient_amplitude.item()

        loss_field_D.backward()  # maximize cost for discriminator
        op_D.step()  # step

        ## train field generator
        op_G.zero_grad()

        G_loss = -1*criterion_wgan(Field_D(fake_field).mean(dim=(-2, -1)))

        consistency_loss = criterion_cycle(consistency_holo, holo) + criterion_cycle(consistency_field, real_field)
        consistency_ssim = (1 - criterion_ssim(real_amplitude, consistency_amplitude)) + (
                    1 - criterion_ssim(real_phase, consistency_phase))
        consistency_loss_distance = criterion_cycle(consistency_distance, real_distance)

        loss_4_gan_x = G_loss + args.cycle_regularizer * consistency_loss \
                       + args.distance_regularizer*consistency_loss_distance + args.ssim_regularizer*consistency_ssim

        loss_sum_G += G_loss.item() + args.cycle_regularizer * consistency_loss.item() + args.ssim_regularizer*consistency_ssim.item()
        loss_sum_D += args.distance_regularizer*consistency_loss_distance.item()

        loss_4_gan_x.backward()
        op_G.step()

        if (it + 1) % args.chk_iter == 0:

            lr_scheduler_G.step()
            lr_scheduler_D.step()

            print("[Iterations : %d/%d] : Generator Loss : %2.4f, Distance Loss : %2.4f, Discriminator Loss(X) : %2.4f, Gradient penalty loss : %2.4f"
                  % (it+1, args.iterations, loss_sum_G / (args.chk_iter), loss_sum_D/(args.chk_iter),
                     loss_sum_Field_D / (args.chk_iter), loss_sum_Field_D_penalty/(args.chk_iter)))

            make_path(saving_path)
            make_path(os.path.join(saving_path, 'generated'))

            # path for saving result
            p = os.path.join(saving_path, 'generated', 'iterations_' + str(it + 1))
            make_path(p)

            loss_field_D_list.append(loss_sum_Field_D)
            loss_G_list.append(loss_sum_G)
            loss_distance_G_list.append(loss_sum_D)
            loss_field_D_penalty_list.append(loss_sum_Field_D_penalty)

            loss_sum_Field_D = 0
            loss_sum_Field_D_penalty = 0
            loss_sum_G = 0
            loss_sum_D = 0

            Field_G.eval()
            distance_G.eval()

            test_gt = iter(DataLoader(test_gt_loader, batch_size=1, shuffle=False))
            test_holo = iter(DataLoader(test_holo_loader, batch_size=1, shuffle=False))

            for b in range(len(test_holo_list)):

                if args.data_name == 'mnist':
                    gt_batch, _ = next(iter(test_gt))

                    if args.mode == 'phase':
                        real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                        real_amplitude = torch.ones_like(real_phase).to(device) * 0.6

                    elif args.mode == 'amplitude':
                        real_amplitude = (1 - F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6
                        real_phase = torch.zeros_like(real_amplitude).to(device)

                    elif args.mode == 'complex_amplitude':
                        real_phase = F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)
                        gt_batch, _ = next(iter(test_gt))
                        real_amplitude = (1 - F.pad(gt_batch, (56, 56, 56, 56), 'constant', 0).to(device)) * 0.6

                    d = (torch.Tensor([test_holo_list[b]]).reshape([1, 1, 1, 1])/ args.distance_normalize) - args.distance_normalize_constant
                    d = d.to(device).float()
                    holo = holo_G(real_amplitude, real_phase, d).detach()

                    real_phase *= args.phase_normalize
                else:
                    real_amplitude, real_phase = next(test_gt)
                    holo = next(test_holo)

                real_amplitude = center_crop(real_amplitude, args.crop_size).to(device).float()
                real_phase = center_crop(real_phase, args.crop_size).to(device)
                holo = center_crop(holo, args.crop_size).to(device).float()

                d = test_holo_list[b]

                ## generate test amplitude and distance
                fake_amplitude, fake_phase = Field_G(holo)
                fake_distance = distance_G(holo)

                fake_holo = holo_G(fake_amplitude, fake_phase, fake_distance).cpu().detach().numpy()[0][0]
                holo = holo.cpu().detach().numpy()[0][0]
                fake_distance = (fake_distance.cpu().detach().numpy()[0][0][0][0] + args.distance_normalize_constant)* args.distance_normalize

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
                plt.title('ground truth' + str(d) + 'mm')
                plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
                plt.axis('off')
                plt.colorbar(shrink=0.75)
                plt.subplot(2, 3, 3)
                plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
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
                plt.title('output phase')
                plt.imshow(fake_phase, cmap='jet', vmax=pi, vmin=0)
                plt.axis('off')
                plt.colorbar(shrink=0.75)

                fig_save_name = os.path.join(p, 'test' + str(b + 1) + '.png')
                fig2.savefig(fig_save_name)
                plt.close(fig2)

    loss = {}
    loss['x_generator_loss'] = loss_G_list
    loss['d_generator_loss'] = loss_distance_G_list
    loss['field_discriminator_loss'] = loss_field_D_list
    loss['field_discriminator_loss_penalty'] = loss_field_D_penalty_list

    save_data = {'Field_G_state_dict': Field_G.state_dict(),
                 'Field_D_state_dict': Field_D.state_dict(),
                 'distance_G_state_dict': distance_G.state_dict(),
                 'loss': loss,
                 'args': args}

    torch.save(save_data, os.path.join(model_saving_path, "model.pth"))