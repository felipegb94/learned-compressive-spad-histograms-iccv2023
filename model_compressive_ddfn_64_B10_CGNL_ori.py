#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting, DeepBoosting
from compressive3D_layer import Compressive3DLayer, Compressive3DLayerWithBias



class LITDeepBoostingCompressive(LITDeepBoosting):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		k=16):
		assert(k > 1), "k needs to be > 1"
		# Init parent class
		super(LITDeepBoostingCompressive, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.k = k
		self.compressive_layer = Compressive3DLayer(k = k)

	def forward(self, x):
		compressive_out = self.compressive_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(compressive_out)
		return out

class LITDeepBoostingCompressiveWithBias(LITDeepBoosting):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		k=16):
		assert(k > 1), "k needs to be > 1"
		# Init parent class
		super(LITDeepBoostingCompressiveWithBias, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.k = k
		self.compressive_layer = Compressive3DLayerWithBias(k = k)

	def forward(self, x):
		compressive_out = self.compressive_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(compressive_out)
		return out



if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 3
	(nr, nc, nt) = (64, 64, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	# Set compression params
	k = 16

	model = LITDeepBoostingCompressive(k=k)

	outputs = model(inputs)

	print("outputs1 shape: {}".format(outputs[0].shape))
	print("outputs2 shape: {}".format(outputs[1].shape))





