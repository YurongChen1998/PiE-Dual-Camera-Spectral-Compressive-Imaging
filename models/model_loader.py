import torch
import os
from collections import OrderedDict
from models.LRNet import Deep_Image_Prior_Network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def network_model_load(rank):
    im_net = Deep_Image_Prior_Network(3, 'reflection',
                            upsample_mode=['nearest', 'nearest', 'bilinear', 'bilinear', 'bilinear'],
                            skip_n33d=64,
                            skip_n33u=64,
                            skip_n11=64,
                            num_scales=2,
                            n_channels=rank).to(device)   

    return [im_net]

