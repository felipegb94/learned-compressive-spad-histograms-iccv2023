#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting
from CSPH1D_layer import CSPH1DLayer



class LITDeepBoostingCSPH1D(LITDeepBoosting):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		k=16,
		num_bins=1024):
		# Init parent class
		super(LITDeepBoostingCSPH1D, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		
		self.csph1D_layer = CSPH1DLayer(k = k, num_bins=num_bins)

	def forward(self, x):
		(zncc_scores, B) = self.csph1D_layer(x)
		# use forward for inference/predictions
		out = self.deep_boosting_model(zncc_scores)
		return out


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	# Set compression params
	k = 16

	model = LITDeepBoostingCSPH1D(k=k, num_bins=nt)

	outputs = model(inputs)

	print("outputs1 shape: {}".format(outputs[0].shape))
	print("outputs2 shape: {}".format(outputs[1].shape))




