#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITPlainDeepBoosting
from CSPH1D_layer import CSPH1DLayer



class LITPlainDeepBoostingCSPH1D(LITPlainDeepBoosting):
	def __init__(self 
		, init_lr = 1e-4
		, p_tv = 1e-5 
		, lr_decay_gamma = 0.9
		, in_channels=1
		, k=16
		, num_bins=1024
		, init = 'HybridGrayFourier'
		, h_irf = None
		):
		# Init parent class
		super(LITPlainDeepBoostingCSPH1D, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.csph1D_layer = CSPH1DLayer(k = k, num_bins = num_bins, init=init, h_irf=h_irf)

	def forward(self, x):
		(zncc_scores, B) = self.csph1D_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(zncc_scores)
		return out

	def forward_test(self, x):
		(zncc_scores, B) = self.csph1D_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(zncc_scores)
		return out, zncc_scores, B


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	simple_hist_input = torch.zeros((2, 1, nt, 32, 32))
	simple_hist_input[0, 0, 100, 0, 0] = 3
	simple_hist_input[0, 0, 200, 0, 0] = 1
	simple_hist_input[0, 0, 50, 0, 0] = 1
	simple_hist_input[0, 0, 540, 0, 0] = 1

	simple_hist_input[1, 0, 300, 0, 0] = 2
	simple_hist_input[1, 0, 800, 0, 0] = 1
	simple_hist_input[1, 0, 34, 0, 0] = 1
	simple_hist_input[1, 0, 900, 0, 0] = 1

	# Set compression params
	k = 16
	model = LITPlainDeepBoostingCSPH1D(k=k, num_bins=nt)

	outputs = model(inputs)

	print("outputs1 shape: {}".format(outputs[0].shape))
	print("outputs2 shape: {}".format(outputs[1].shape))

	## Test with simple inputs
	out, zncc, B = model.forward_test(simple_hist_input)
	print("Input Shape: {}".format(simple_hist_input.shape))
	print("ZNCC Out Shape: {}".format(zncc.shape))
	print("B Out Shape: {}".format(B.shape))
	
	# Look at outputs
	plt.clf()
	plt.plot(simple_hist_input[0,0,:,0,0], '--', label="Inputs 2")
	plt.plot(zncc[0,0,:,0,0], label='ZNCC Outputs 2')
	plt.plot(simple_hist_input[1,0,:,0,0], '--', label="Inputs 3")
	plt.plot(zncc[1,0,:,0,0], label='ZNCC Outputs 3')
	plt.legend()





