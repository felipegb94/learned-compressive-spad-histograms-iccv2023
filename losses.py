# This is the file for loss functions
import torch
import torch.nn as nn


###########################################
# the set of loss functions
criterion_GAN = nn.MSELoss()
criterion_KL = nn.KLDivLoss()

# inpt, target: [batch_size, 1, h, w]
criterion_L1 = nn.L1Loss()

def criterion_TV(inpt):
    return torch.sum(torch.abs(inpt[:, :, :, :-1] - inpt[:, :, :, 1:])) + \
           torch.sum(torch.abs(inpt[:, :, :-1, :] - inpt[:, :, 1:, :]))

def criterion_L2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))