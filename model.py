##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import torch
from func import *
from models.model_loader import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def admm_denoise(meas, Phi, data_truth, ref_img, ref_img_orig, args):
    #-------------- Initialization --------------#
    x0 = At(meas, Phi)
    ssim_all, psnr_all = [], []
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x, theta, theta2 = x0.to(device), x0.to(device), x0.to(device)
    b1, b2 = torch.zeros_like(x0).to(device), torch.zeros_like(x0).to(device)

    gamma_1 = 0.01
    gamma_2 = 0.01
    iter_num = 1000
    best_psrn = 30

    # ---------------- Iteration ----------------#
    for iter_ in range(args.iter_max):
        theta, theta2 = theta.to(device), theta2.to(device)
        b1, b2 = b1.to(device), b2.to(device)
        
        # --------------- Updata X ---------------#
        c = (gamma_1*(theta+b1) + gamma_2*(theta2+b2)) / (gamma_1 + gamma_2)
        meas_b = A(c, Phi)
        x = c + At((meas - meas_b)/(Phi_sum + (gamma_1 + gamma_2)), Phi)
        x1 = shift_back(x-b1, step=2)
        
        # --------------- Updata V ---------------#
        theta = TV_minimization(x1, ref_img, args.tv_weight, args.tv_iter_max, args.alpha)
        theta = shift(theta, step=2)
        b1 = b1 - (x.to(device) - theta.to(device))
            
        # --------------- Updata Z ---------------#
        x2 = shift_back(x-b2, step=2)
        theta2 = Decomposition_PnP(data_truth, x2, meas, Phi, ref_img_orig, iter_num)

        # --------------- Evaluation ---------------#
        ssim_all = calculate_ssim(data_truth*0.9, theta2)
        psnr_all = calculate_psnr(data_truth*0.9, theta2)
        print('Iter ', iter_,
              'PSNR {:2.2f} dB.'.format(psnr_all),
              'SSIM:{:2.3f}.'.format(ssim_all))
        if psnr_all > best_psrn:
            best_psrn = psnr_all
            best_theta = shift(theta2, step=2)
                
        theta2 = shift(theta2, step=2)
        b2 = b2 - (x.to(device) - theta2.to(device))
    return best_theta, best_psrn



def Decomposition_PnP(data_truth, temp_theta, meas, Phi, ref_img, iter_num):
    best_loss = float('inf')
    H, W, B = data_truth.shape
    loss_fn = torch.nn.L1Loss().to(device)
    im_net = network_model_load(B)
    
    if os.path.exists('result/model_weights_init.pth'):
        im_net[0].load_state_dict(torch.load('result/model_weights_init.pth'))
        os.remove('result/model_weights_init.pth')
        print('----------------------- Load init model weights -----------------------')
        
    if os.path.exists('result/model_weights.pth'):
        im_net[0].load_state_dict(torch.load('result/model_weights.pth'))
        iter_num = 200
        print('----------------------- Load model weights -----------------------')
    
    truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0).to(device)
    temp_theta = temp_theta.permute(2, 0, 1).unsqueeze(0).to(device)
    ref_img = ref_img.permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    im_net[0].train()
    net_params = list(im_net[0].parameters())
    optimizer = torch.optim.Adam(lr=1e-3, params=net_params)
    
    for idx in range(iter_num):
        model_out = im_net[0](ref_img)
        loss = loss_fn(temp_theta.float().to(device), model_out.float().to(device)) 
        loss += loss_fn(meas, A(shift(model_out.squeeze(0).permute(1, 2, 0), 2).to(device), Phi.to(device))) 
        loss_tv = 1*stv_loss(model_out)
        loss += loss_tv
        
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = idx
            best_hs_recon = model_out.detach()
            torch.save(im_net[0].state_dict(), 'result/model_weights.pth')
        
        if (idx+1)%100==0:
            PSNR = calculate_psnr(0.9*truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.4f}, tv_loss:{:.4f}, PSNR:{:.2f}'.format(idx+1, loss.detach().cpu().numpy(), loss_tv.detach().cpu().numpy(), PSNR.detach().cpu().numpy()))
    return best_hs_recon.squeeze(0).permute(1, 2, 0)
