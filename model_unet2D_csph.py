#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_base_spad_lit import LITBaseSPADModel
from csph_layers import CSPH1DGlobalEncodingLayer
from model_unet import Unet
from layers_parametric1D import Gaussian1DLayer, IRF1DLayer



class LITUnet2DCSPH1D(LITBaseSPADModel):
	def __init__(self 
		, init_lr = 1e-4
		, p_tv = 1e-5 
		, lr_decay_gamma = 0.9
		, k=16
		, num_bins=1024
		, init = 'HybridGrayFourier'
		, h_irf = None
		, optimize_csph = False
		):
		unet_model = Unet(
						in_channels=k, 
						out_channels=1, 
						nf0=64, 
						num_down=3, 
						max_channels=512, 
						upsampling_mode='nearest', 
						use_dropout=False, 
						outermost_linear=False)

		# Init parent class
		super(LITUnet2DCSPH1D, self).__init__(
			init_lr=init_lr, 
			p_tv=p_tv, 
			lr_decay_gamma=lr_decay_gamma,
			backbone_net=unet_model,
			data_loss_id = 'L1',
			)

		self.csph1D_layer = CSPH1DGlobalEncodingLayer(k = k, num_bins = num_bins, init = init, h_irf = h_irf, optimize_weights=optimize_csph)

		## Replace the hist reconstruction layer
		gauss_layer = Gaussian1DLayer(gauss_len=num_bins, out_dim=-3)
		if(h_irf is None):
			self.hist_rec_layer = torch.nn.Sequential(
				gauss_layer
			)
		else:
			irf_layer = IRF1DLayer(h_irf, conv_dim=0)
			self.hist_rec_layer = torch.nn.Sequential(
				gauss_layer, 
				irf_layer
			)

	def forward(self, x):
		B = self.csph1D_layer(x.squeeze(1))
		# use forward for inference/predictions
		out_depths = self.backbone_net(B)
		out_hist = self.hist_rec_layer(out_depths)
		return (out_hist.squeeze(1), out_depths)


if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters

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
	model = LITUnet2DCSPH1D(k=k, num_bins=nt)
	outputs = model(inputs)
	print("LITUnet2DCSPH1D model: {} params".format(count_parameters(model)))
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))

	# ## Test with simple inputs
	# out, zncc, B = model.forward_test(simple_hist_input)
	# print("Input Shape: {}".format(simple_hist_input.shape))
	# print("ZNCC Out Shape: {}".format(zncc.shape))
	# print("B Out Shape: {}".format(B.shape))
	
	# # Look at outputs
	# plt.clf()
	# plt.plot(simple_hist_input[0,0,:,0,0], '--', label="Inputs 2")
	# plt.plot(zncc[0,0,:,0,0], label='ZNCC Outputs 2')
	# plt.plot(simple_hist_input[1,0,:,0,0], '--', label="Inputs 3")
	# plt.plot(zncc[1,0,:,0,0], label='ZNCC Outputs 3')
	# plt.legend()





