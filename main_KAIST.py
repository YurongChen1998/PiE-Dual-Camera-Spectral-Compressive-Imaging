##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import time
import torch
import shutil
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import admm_denoise
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)


def main(data_name, args):
    #----------------------- Data Configuration -----------------------#
    if os.path.exists('./model_weights_init.pth'):
        shutil.copyfile('./model_weights_init.pth', 'result/model_weights_init.pth')

    dataset_dir = './Dataset/KAIST_Dataset/Orig_data/'
    results_dir = './result/KAIST/' + data_name + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    matfile = dataset_dir + data_name + '.mat'
    h, w, nC, step, beta = 256, 256, 28, 2, 0.9
    data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])

    data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC))
    for i in range(nC):
        data_truth_shift[:, i*step:i*step+h, i] = data_truth[:, :, i]
    data_truth_shift = data_truth_shift * beta

    # RGB_measurement or Bayer_RGB
    ref_matfile = './Dataset/KAIST_Dataset/Bayer_RGB/' + data_name + '_rgb.mat'
    ref_img = torch.from_numpy(sio.loadmat(ref_matfile)['rgb_recon']).to(device)
    ref_img_1 = torch.unsqueeze(ref_img[:, :, 2], 2)
    ref_img_2 = torch.unsqueeze(ref_img[:, :, 1], 2)
    ref_img_3 = torch.unsqueeze(ref_img[:, :, 0], 2)
    ref_img_orig = torch.cat((ref_img_1, ref_img_2, ref_img_3), dim=2)
    ref_img = F.interpolate(ref_img_orig, size = [nC], mode='linear')
    ref_img = ref_img * (1 - beta) 
    #------------------------------------------------------------------#


    #----------------------- Mask Configuration -----------------------#
    mask = torch.zeros((h, w + step*(nC - 1)))
    mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
    mask_256 = torch.from_numpy(sio.loadmat('./Dataset/KAIST_Dataset/mask256.mat')['mask'])
    for i in range(nC):
        mask_3d[:, i*step:i*step+h, i] = mask_256
    Phi = mask_3d
    meas = torch.sum(Phi * data_truth_shift, 2)

    plt.figure()
    plt.imshow(meas,cmap='gray')
    plt.savefig(results_dir+'meas.png')
    #------------------------------------------------------------------#


    begin_time = time.time()
    recon, best_psrn = admm_denoise(meas.to(device), Phi.to(device), data_truth.to(device), ref_img.to(device), ref_img_orig.to(device), args)
    end_time = time.time()
    recon = shift_back(recon, step=2)
    print('=========== Running time {:.1f} seconds.'.format(end_time - begin_time))
    sio.savemat(results_dir+'result_'+str(best_psrn.item())+'.mat', {'img':recon.cpu().numpy()})


if __name__=='__main__':
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_max', default=7, help="Maximum number of iterations")
    parser.add_argument('--tv_weight', default=30, help="weight of PIDS")
    parser.add_argument('--tv_iter_max', default=30, help="iterations of PIDS")
    parser.add_argument('--alpha', default=0.65, help="level of PIDS")
    args = parser.parse_args()
    #------------------------------------------------------------------#

    scene_idx = ['01'] #, '02', '03', '04', '05', '06', '07', '08', '09', '10'
    for idx in scene_idx:
        if idx == '01' and '03':
            args.tv_weight, args.tv_iter_max, args.alpha = 30, 30, 0.65
        elif idx == '02':
            args.tv_weight, args.tv_iter_max, args.alpha = 45, 45, 0.55
        elif idx == '04':
            args.tv_weight, args.tv_iter_max, args.alpha = 24, 24, 0.50
        elif idx == '05':
            args.tv_weight, args.tv_iter_max, args.alpha = 35, 30, 0.85
        elif idx == '06':
            args.tv_weight, args.tv_iter_max, args.alpha = 35, 30, 0.95
        elif idx == '07':
            args.tv_weight, args.tv_iter_max, args.alpha = 45, 40, 0.45
        elif idx == '08':
            args.tv_weight, args.tv_iter_max, args.alpha = 27, 30, 0.45
        elif idx == '09':
            args.tv_weight, args.tv_iter_max, args.alpha = 27, 30, 0.70
        elif idx == '10':
            args.tv_weight, args.tv_iter_max, args.alpha = 27, 30, 0.80

        main('scene' + idx, args)
        if os.path.exists('result/model_weights.pth'):
            os.remove('result/model_weights.pth')
