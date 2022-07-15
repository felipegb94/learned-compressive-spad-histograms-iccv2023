#### Standard Library Imports

#### Library imports
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import pytorch_lightning as pl

from IPython.core import debugger
breakpoint = debugger.set_trace
import torchio as tio

#### Local imports


if __name__=='__main__':
	pl.seed_everything(2)
	## Generate inputs
	k=7
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024)
	inputs2D = torch.randn((batch_size, 1, nr, nc))
	kernel2D = (4, 4)
	inputs3D = torch.randn((batch_size, 1, nt, nr, nc))
	kernel3D = (1024, 4, 4)

	patches = inputs3D.unfold(-3, kernel3D[-3], kernel3D[-3]).unfold(-2, kernel3D[-2], kernel3D[-2]).unfold(-1, kernel3D[-1], kernel3D[-1])
	unfold_shape = patches.size()
	patches = patches.contiguous().view(batch_size, -1, kernel3D[-3], kernel3D[-2], kernel3D[-1])
	print(patches.shape)