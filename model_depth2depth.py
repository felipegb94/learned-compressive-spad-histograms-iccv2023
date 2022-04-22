#### Standard Library Imports
import os

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from losses import criterion_L2, criterion_KL, criterion_TV
import tof_utils
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting



class LITDeepBoostingDepth2Depth(LITDeepBoosting):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		# Init parent class
		super(LITDeepBoostingDepth2Depth, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		
	def get_input_data(self, sample):
		return sample["est_bins_argmax"]
	