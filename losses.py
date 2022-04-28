#### Standard Library Imports

#### Library imports
import torch
import torch.nn as nn

#### Local imports


###########################################
# the set of loss functions
criterion_GAN = nn.MSELoss()

# We use mean reduction even if it is mathematically incorrect. This is what was used in Peng et al., 2020 and Lindell et al., 2018
# One reason we may prefer mean of batchmean is that if the histogram image increases in dimension then we want the kldiv loss to stay
# in the same order of magnitude
# NOTE: The KLDiv loss expects input to be log-probabilities, but NOT the target
# criterion_KL = nn.KLDivLoss()
criterion_KL = nn.KLDivLoss(reduction='mean') 
# criterion_KL = nn.KLDivLoss(reduction='batchmean')

# inpt, target: [batch_size, 1, h, w]
criterion_L1 = nn.L1Loss()

def criterion_TV(inpt):
    return torch.sum(torch.abs(inpt[:, :, :, :-1] - inpt[:, :, :, 1:])) + \
           torch.sum(torch.abs(inpt[:, :, :-1, :] - inpt[:, :, 1:, :]))

def criterion_L2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))